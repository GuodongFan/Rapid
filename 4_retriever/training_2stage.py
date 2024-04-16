"""
This examples trains a CrossEncoder for the STSbenchmark task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it output a continious labels 0...1 to indicate the similarity between the input pair.
It does NOT produce a sentence embedding and does NOT work for individual sentences.
Usage:
python training_stsbenchmark.py
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers import InputExample
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
from beir.datasets.data_loader import GenericDataLoader
import argparse
from dataset import *
import torch


class MyCrossEncoder(CrossEncoder):
    def __init__(self, model_name:str, num_labels:int = None, max_length:int = None, device:str = None, tokenizer_args:Dict = {},
                  automodel_args:Dict = {}, default_activation_function = None,
                 hard_negative_dataset = None,
                 hard_negative = False):
        super().__init__(model_name, num_labels, max_length, device, tokenizer_args,
                  automodel_args, default_activation_function)

        self.hard_negative_dataset=hard_negative_dataset
        self.add_hard_negative = hard_negative
    def smart_batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []
        num_texts = len(batch[0].texts)
        batch_len = len(batch)
        if batch_len == 0:
            return None, None

        # 正样本
        for example_idx, example in enumerate(batch):

            for idx, text in enumerate(example.texts):
                if idx % 2 == 0:
                    query = text.strip()
                    texts[idx].append(query)
                else:
                    texts[idx].append(text.get('text').strip())

            labels.append(1)

            # add 1 negative
            neg_ids = list(range(batch_len))
            neg_ids.remove(example_idx)
            neg_id = random.choice(neg_ids)
            neg_code = batch[neg_id].texts[1].get('text').strip()
            texts[0].append(query)
            texts[1].append(neg_code)
            labels.append(0)


            # add 1 hard negative
            if self.add_hard_negative:
                item = self.hard_negative_dataset.get_queryid(example.guid[0])

                hard_example = self.hard_negative_dataset.__getitem__(item)
                for idx, text in enumerate(hard_example.texts):
                    if idx % 2 == 0:
                        texts[idx].append(query)
                    else:
                        texts[idx].append(text.get('text').strip())

                labels.append(0)

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels


def main(generated_path, prefix, output_path, hard_negative, lang):
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    logger = logging.getLogger(__name__)
    #### /print debug information to stdout

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    generated_path = "{}{}".format(generated_path, lang)

    print('Generated Path: {}'.format(generated_path))

    valid_path = "{}/valid/".format(generated_path)

    print('Valid Path: {}'.format(generated_path))

    output_path = "{}stage2_{}".format(output_path, lang)

    if hard_negative:
        output_path = "{}_{}".format(output_path, "hard")

    print('Output Path: {}'.format(output_path))

    model_name = './models/mlm_model_{}'.format(lang)

    #Define our Cross-Encoder
    train_batch_size = 10
    num_epochs = 1

    corpus, gen_queries, gen_qrels = GenericDataLoader(
        generated_path, prefix=prefix
    ).load(split="train")

    fpath_hard_negatives = os.path.join(generated_path, "hard-negatives.jsonl")
    hard_negative_dataset = HardNegativeDataset(
        fpath_hard_negatives, gen_queries, corpus
    )

    train_samples = []
    dev_samples = []
    test_samples = []

    for qid, cidic in gen_qrels.items():
        query = gen_queries.get(qid)
        for cid, _ in cidic.items():
            pass
        if cid == None:
            continue
        code = corpus.get(cid)
        train_samples.append(InputExample(texts=[query, code], guid=[qid, cid], label=1))


    corpus, gen_queries, gen_qrels = GenericDataLoader(
        valid_path, prefix=''
    ).load(split="valid")


    for qid, cidic in gen_qrels.items():
        query = gen_queries.get(qid)
        for cid, _ in cidic.items():
            pass
        if cid == None:
            continue
        code = corpus.get(cid)
        dev_samples.append(InputExample(guid=[qid, cid], texts=[query, code.get('text')], label=1))

        # add 1 negative
        dev_num = len(gen_qrels)
        neg_ids = list(gen_queries)
        while True:
            neg_id = random.choice(neg_ids)
            neg_code = gen_qrels.get(neg_id)
            for neg_cid, _ in neg_code.items():
                pass
            if neg_cid != cid:
                break

        dev_samples.append(InputExample(guid=[qid, neg_cid], texts=[query, corpus.get(neg_cid).get('text')], label=0))


    # We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)


    # We add an evaluator, which evaluates the performance during training
    evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='sts-dev')


    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
    logger.info("Warmup-steps: {}".format(warmup_steps))

    #We use distilroberta-base as base model and set num_labels=1, which predicts a continous score between 0 and 1
    model = MyCrossEncoder(model_name, num_labels=1, hard_negative_dataset=hard_negative_dataset, hard_negative=hard_negative)

    # Train the model
    model.fit(train_dataloader=train_dataloader,
              evaluator=evaluator,
              evaluation_steps=1000,
              epochs=num_epochs,
              warmup_steps=warmup_steps,
              output_path=output_path)

if __name__ == "__main__":
    # --generated_path Dataset/SQL/ --prefix qgen --output_path ./output/output_2stage
    # --generated_path Dataset/SQL/groundtruth --output_path ./output/output_2stage_supervised
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_path", default='./Dataset/')
    parser.add_argument("--output_path", default='./models/')
    parser.add_argument("--prefix", default='qgen')
    parser.add_argument("--lang", default='Solidity')
    parser.add_argument("--hard_negative", action='store_true', default=True)
    args = parser.parse_args()

    main(args.generated_path, args.prefix, args.output_path, args.hard_negative, args.lang)
