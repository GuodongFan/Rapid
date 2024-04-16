import sys
from beir.datasets.data_loader import GenericDataLoader
from generate import QueryGenerator as QGen
import os
import argparse

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch, logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class QGenModel:
    def __init__(self, model_path: str, gen_prefix: str = "", use_fast: bool = True, device: str = None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.gen_prefix = gen_prefix
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Use pytorch device: {}".format(self.device))
        self.model = self.model.to(self.device)

    def generate(self, corpus: List[Dict[str, str]], ques_per_passage: int, top_k: int, max_length: int,
                 top_p: float = None, temperature: float = None) -> List[str]:

        texts = [(self.gen_prefix + doc["title"] + " " + doc["text"]) for doc in corpus]
        encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        # Top-p nucleus sampling
        # https://huggingface.co/blog/how-to-generate
        with torch.no_grad():
            input_ids = encodings['input_ids'].to(self.device)
            if not temperature:
                outs = self.model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    max_length=max_length,  # 64
                    top_k=top_k,  # 25
                    top_p=top_p,  # 0.95
                    num_return_sequences=ques_per_passage,  # 1
                    return_dict_in_generate=True, output_scores=True
                )
            else:
                outs = self.model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    max_length=max_length,  # 64
                    top_k=top_k,  # 25
                    temperature=temperature,
                    num_return_sequences=ques_per_passage  # 1
                )

        # only use id's that were generated
        # gen_sequences has shape [3, 15]
        gen_sequences = outs.sequences[:, max_length:]
        # let's stack the logits generated at each step to a tensor and transform
        # logits to probs
        probs = torch.stack(outs.scores, dim=1).softmax(-1)  # -> shape [3, 15, vocab_size]

        # now we need to collect the probability of the generated token
        # we need to add a dummy dim in the end to make gather work
        gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)

        return self.tokenizer.batch_decode(outs.sequences, skip_special_tokens=True)

def qgen(
    lang,
    qgen_prefix="t5",
    ques_per_passage=3,
    generator_name_or_path='./models/pre-trained/codet5-base-multi-sum/',
    #generator_name_or_path='/data1/fgd/workplace/CodeT5-main/sh/saved_models/summarize/sql/codet5_base_multi_sum_1000_lr5_bs10_src128_trg32_pat2_e1/checkpoint-best-bleu', #"./models/pre-trained/codet5-base-multi-sum/", #"Salesforce/codet5-base-multi-sum",./models/prompt_t5/ /data1/fgd/workplace/CodeT5-main/sh/saved_models/summarize/sql/codet5_base_multi_sum_all_lr5_bs10_src128_trg32_pat2_e15/checkpoint-best-bleu
    bsz=5
):
    ROOT_PATH = './Dataset/'

    data_path = '{}{}/'.format(ROOT_PATH, lang)
    output_dir = '{}{}/'.format(ROOT_PATH, lang)
    print('Data path: {}'.format(data_path))
    print('Out path: {}'.format(output_dir))
    print('Prefix: {}'.format(qgen_prefix))
    print('Model path: {}'.format(generator_name_or_path))
    #### Provide the data_path where nfcorpus has been downloaded and unzipped
    corpus = GenericDataLoader(data_path).load_corpus()

    #### question-generation model loading
    generator = QGen(model=QGenModel(generator_name_or_path))

    #### Query-Generation using Nucleus Sampling (top_k=25, top_p=0.95) ####
    #### https://huggingface.co/blog/how-to-generate
    #### Prefix is required to seperate out synthetic queries and qrels from original
    prefix = qgen_prefix

    #### Generating 3 questions per passage.
    #### Reminder the higher value might produce lots of duplicates
    #### Generate queries per passage from docs in corpus and save them in data_path
    try:
        generator.generate(
            corpus,
            output_dir=output_dir,
            ques_per_passage=ques_per_passage,
            prefix=prefix,
            batch_size=bsz,
            max_length=32,
        )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            raise RuntimeError(
                f"CUDA out of memory during query generation "
                f"(queries_per_passage: {ques_per_passage}, batch_size_generation: {bsz}). "
                f"Please try smaller `queries_per_passage` and/or `batch_size_generation`."
            )

    if not os.path.exists(os.path.join(output_dir, "corpus.jsonl")):
        os.system(f"cp {data_path}/corpus.jsonl {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='Solidity')
    parser.add_argument("--prefix", default='qgen')
    parser.add_argument("--ques_per_passage", default=3, type=int)
    args = parser.parse_args()
    qgen(args.lang, args.prefix, args.ques_per_passage)
