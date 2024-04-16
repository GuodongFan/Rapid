from tqdm.autonotebook import trange
from util import write_to_json, write_to_tsv
from typing import Dict
import logging, os
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
from beir.generation.models import QGenModel
from transformers.models.t5 import T5ForConditionalGeneration, T5Tokenizer,T5Config
from transformers import RobertaTokenizer
from openprompt import PromptDataLoader, PromptForGeneration
from openprompt.plms import T5TokenizerWrapper
from openprompt.prompts import PrefixTuningTemplate,SoftTemplate
from openprompt.data_utils import InputExample
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Dict
from tqdm import tqdm

import sys

sys.path.append("./")

from metrics.bert_perplexity_sentence import Perplexity_Checker

logger = logging.getLogger(__name__)

# 模型名字
MODEL_NAME = "bert-base-uncased"

text_formatter = lambda x: "[CLS] {} [SEP]".format(x)


# pchecker = Perplexity_Checker(MODEL_NAME=MODEL_NAME, device='cuda')

class QGenPModel():
    def __init__(self, model_path: str, gen_prefix: str = "", use_fast: bool = True, device: str = None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)
        self.plm = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.mytemplate = PrefixTuningTemplate(model=self.plm, tokenizer=self.tokenizer,
                                          text='Code: {"placeholder":"text_a"} {"special": "<eos>"} Summarization: {"mask"} ',
                                          using_decoder_past_key_values=False)
        self.model = PromptForGeneration(plm=self.plm,template=self.mytemplate, freeze_plm=False, tokenizer=self.tokenizer, plm_eval_mode=False)
        self.model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"))) # 不加这个不行呢
        self.gen_prefix = gen_prefix
        self.wrapperClass = T5TokenizerWrapper
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Use pytorch device: {}".format(self.device))
        self.model = self.model.to(self.device)


def get_bleu(recover, reference):
    return sentence_bleu([reference.split()], recover.split(), smoothing_function=SmoothingFunction().method4, )



generation_arguments = {
    "max_length": 20,
    "temperature": 1.0,
    "top_k": 25,
    "top_p": 0.95,
    "num_beams": 20,
}

class QueryGenerator:
    def __init__(self, model, **kwargs):
        self.model = model
        self.qrels = {}
        self.queries = {}

    @staticmethod
    def save(output_dir: str, queries: Dict[str, str], qrels: Dict[str, Dict[str, int]], prefix: str):

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, prefix + "-qrels"), exist_ok=True)

        query_file = os.path.join(output_dir, prefix + "-queries.jsonl")
        qrels_file = os.path.join(output_dir, prefix + "-qrels", "train.tsv")

        logger.info("Saving Generated Queries to {}".format(query_file))
        write_to_json(output_file=query_file, data=queries)

        logger.info("Saving Generated Qrels to {}".format(qrels_file))
        write_to_tsv(output_file=qrels_file, data=qrels)

    def generate(self,
                 corpus: Dict[str, Dict[str, str]],
                 output_dir: str,
                 top_p: int = 0.95,
                 top_k: int = 25,
                 max_length: int = 64,
                 ques_per_passage: int = 1,
                 prefix: str = "gen",
                 batch_size: int = 5,
                 save: bool = True,
                 save_after: int = 100000):

        logger.info(
            "Starting to Generate {} Questions Per Passage using top-p (nucleus) sampling...".format(ques_per_passage))
        logger.info("Params: top_p = {}".format(top_p))
        logger.info("Params: top_k = {}".format(top_k))
        logger.info("Params: max_length = {}".format(max_length))
        logger.info("Params: ques_per_passage = {}".format(ques_per_passage))
        logger.info("Params: batch size = {}".format(batch_size))

        count = 0
        corpus_ids = list(corpus.keys())
        corpus = [corpus[doc_id] for doc_id in corpus_ids]
        test_data = []
        for code in corpus:
            test_data.append(InputExample(text_a=code.get('text'), text_b=''))

        #import random
        #sample_num = 100
        #random.seed(1024)
        #test_data = random.sample(test_data, min(sample_num, len(test_data)))

        test_dataloader = PromptDataLoader(dataset=test_data, template=self.model.mytemplate, tokenizer=self.model.tokenizer,
    tokenizer_wrapper_class=self.model.wrapperClass, max_seq_length=128, decoder_max_length=32,
    batch_size=batch_size,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

        for step, inputs in enumerate(tqdm(test_dataloader)):
            inputs = inputs.cuda()
            _, queries = self.model.model.generate(inputs, num_return_sequences=ques_per_passage, **generation_arguments)
            size = len(inputs.input_ids)

            assert len(queries) == size * ques_per_passage

            start_idx = step * batch_size
            for idx in range(size):
                # Saving generated questions after every "save_after" corpus ids
                if (len(self.queries) % save_after == 0 and len(self.queries) >= save_after):
                    logger.info("Saving {} Generated Queries...".format(len(self.queries)))
                    self.save(output_dir, self.queries, self.qrels, prefix)

                corpus_id = corpus_ids[start_idx + idx]
                start_id = idx * ques_per_passage
                end_id = start_id + ques_per_passage
                query_set = set([q.strip() for q in queries[start_id:end_id]])

                for query in query_set:
                    count += 1
                    query_id = "genQ" + str(count)
                    self.queries[query_id] = query
                    self.qrels[query_id] = {corpus_id: 1}

        # Saving finally all the questions
        logger.info("Saving {} Generated Queries...".format(len(self.queries)))
        self.save(output_dir, self.queries, self.qrels, prefix)

