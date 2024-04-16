"""
This is an example how to train with sentence-transformers.

It trains the model just for 2k steps using equal weighting of all provided dataset files.

Run:
python training.py exp-name file1.jsonl.gz [file2.jsonl.gz] ...

"""
import math
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, InformationRetrievalEvaluator
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
import os
from beir.datasets.data_loader import GenericDataLoader
from torch.utils.data import DataLoader
from torch import nn, Tensor, device
import torch
from dataset import *
import argparse
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)
from InputExample import InputExample


rel = 'rel'
gen = 'gen'
all = 'all'
choices = [rel, gen, all]

# Define a function to validate the input value
def enum_type(val):
    if val not in choices:
        raise argparse.ArgumentTypeError(f"{val} is not a valid choice.")
    return val

class MySentenceTransformer(SentenceTransformer):
    def __init__(self, model_name_or_path: Optional[str] = None,
                 modules: Optional[Iterable[nn.Module]] = None,
                 device: Optional[str] = None,
                 cache_folder: Optional[str] = None,
                 use_auth_token: Union[bool, str, None] = None,
                 hard_negative_dataset = None,
                 hard_negative = False,
                 ):
        super().__init__(model_name_or_path,
                 modules,
                 device,
                 cache_folder,
                 use_auth_token)

        self.hard_negative_dataset=hard_negative_dataset
        self.add_hard_negative = hard_negative
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<encoder-only>"]})

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0].texts)

        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)
            labels.append(example.label)

            # add hard negative
            if self.add_hard_negative:
                for i in range(1):
                    item = self.hard_negative_dataset.get_queryid(example.guid[0])
                    if item == None:
                        print('Item is None')
                        print(example.guid[0])
                    example = self.hard_negative_dataset.__getitem__(item)
                    for idx, text in enumerate(example.texts):
                        texts[idx].append(text)
                    labels.append(example.label)

        labels = torch.tensor(labels)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.tokenize(texts[idx])
            sentence_features.append(tokenized)

        return sentence_features, labels

def main(generated_path, prefix, output_path, hard_negative, lang, retrievers, train_type):
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    print('************************')
    print('prefix {} \nlang {} \nhard {} \nretrievers {} \nsupervised {}'.format(prefix, lang, hard_negative, retrievers, train_type))


    generated_path = "{}{}".format(generated_path, lang)

    print('Working Path: {}'.format(generated_path))

    train_path = "{}/train/".format(generated_path)
    valid_path = "{}/valid/".format(generated_path)
    test_path = "{}/test/".format(generated_path)
    hard_path = generated_path

    print('Valid Path: {}'.format(valid_path))

    #model_name = './models/mlm_model_{}'.format(lang)

    # model_name = './models/tsdae_{}_model'.format(lang)

    #model_name = 'bert-base-uncased'

    #model_name = 'distilbert-base-uncased'

    model_name = '/data1/fgd/workplace/models/distilbert' #distilbert-base-uncased
    #model_name = '/data1/fgd/workplace/ZDAN/models/tsdae_SQL_model/'

    #model_name = '/data1/fgd/workplace/models/graphcodebert-base'
    #model_name = '/data1/fgd/workplace/models/unixcoder-base/'

    hard_negative_file = 'hard-negatives.jsonl'
    if hard_negative:
        output_path = "{}stage1_{}_{}".format(output_path, lang, "hard")
        for retriever in retrievers:
            output_path = "{}_{}".format(output_path, retriever)
    else:
        output_path = "{}stage1_{}".format(output_path, lang)

    if train_type == rel:
        output_path = "{}_{}".format(output_path, 'rel')
        hard_path = train_path
    elif train_type == all:
        #output_path = "{}_{}".format(output_path, 'all')
        hard_negative_file = 'hard-negatives-all.jsonl'
        #model_name = output_path
        output_path = "{}_{}".format(output_path, 'all')
        hard_path = generated_path


    print('Load Model: {}'.format(model_name))

    print('Hard Path: {}'.format(hard_path))
    print('Hard Negative File: {}'.format(hard_negative_file))
    print('Output Model Path: {}'.format(output_path))
    print('************************')

    batch_size_pairs = 128
    batch_size_triplets = 128
    steps_per_epoch = 5000 # 不设置就自动根据 num/batchsize得出

    num_epochs = 1
    max_seq_length = 256
    use_amp = True


    # 整合数据
    if train_type == gen:
        corpus, gen_queries, gen_qrels = GenericDataLoader(
            generated_path, prefix=prefix
        ).load(split="train")
        print('real')
    elif train_type == rel:
        corpus, gen_queries, gen_qrels = GenericDataLoader(
            train_path, prefix=''
        ).load(split="train")
    else:
        corpus, gen_queries, gen_qrels = GenericDataLoader(
            generated_path, prefix=prefix
        ).load(split="train")

        #gen_queries = {}
        #gen_qrels = {}
        corpus, queries, qrels = GenericDataLoader(
            train_path, prefix=''
        ).load(split="train")

        gen_queries.update(queries)
        gen_qrels.update(qrels)

    fpath_hard_negatives = os.path.join(hard_path, hard_negative_file)
    hard_negative_dataset = HardNegativeDataset(
        fpath_hard_negatives, gen_queries, corpus, retrievers
    )

    train_examples  = []

    for qid, cidic in gen_qrels.items():
        query = gen_queries.get(qid)
        for cid, _ in cidic.items():

            code = corpus.get(cid)
            train_examples.append(InputExample(texts=[query, code], guid=[qid, cid]))


    # Special data loader to load from multiple datasets
    train_dataloader = DataLoader(train_examples, shuffle=True,
                                              batch_size=10)

    corpus, gen_queries, gen_qrels = GenericDataLoader(
        valid_path, prefix=''
    ).load(split="valid")

    test_examples  = []
    work_idx = 0
    for qid, cidic in gen_qrels.items():
        query = gen_queries.get(qid)
        for cid, _ in cidic.items():
            pass
        if cid == None:
            continue
        code = corpus.get(cid)
        test_examples.append(InputExample(guid=str(work_idx), texts=[query, code]))

        work_idx += 1


    print('SentenceTransformer model begin')
    ## SentenceTransformer model
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

    # Define a function for custom initialization
    def init_weights(module):
        if isinstance(module, (torch.nn.Linear,)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (torch.nn.Embedding,)):
            module.weight.data.normal_(mean=0.0, std=0.02)

    # Apply the custom initialization function to the model's parameters
    #word_embedding_model.apply(init_weights)

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    norm = models.Normalize()
    model = MySentenceTransformer(modules=[word_embedding_model, pooling_model, norm], hard_negative_dataset=hard_negative_dataset, hard_negative=hard_negative)

    #
    evaluator = InformationRetrievalEvaluator(gen_queries, corpus, gen_qrels)

    # Our training loss
    train_loss = losses.MultipleNegativesSymmetricRankingLoss(model, scale=20, similarity_fct=util.dot_score)
    #train_loss = losses.MultipleNegativesRankingLoss(model, scale=20, similarity_fct=util.dot_score)

    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              evaluation_steps=2000,
              epochs=1,
              warmup_steps=warmup_steps,
              steps_per_epoch=None,
              scheduler='warmupconstant',   #Remove this line when you train on larger datasets. After warmup, LR will be constant
              use_amp=use_amp
              )


    model.save(output_path) # 'A./output_1stage'

    corpus, test_queries, test_qrels = GenericDataLoader(
        test_path, prefix=''
    ).load(split="test")

    test_evaluator = InformationRetrievalEvaluator(test_queries, corpus, test_qrels)

    print(model.evaluate(test_evaluator))


if __name__ == "__main__":

    # --generated_path Dataset/SQL/groundtruth --output_path ./output/output_1stage_supervised
    # --generated_path Dataset/SQL/ --prefix qgen --output_path ./output/output_1stage
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_path", default='./Dataset/')
    parser.add_argument("--output_path", default='./models/')
    parser.add_argument("--prefix", default='qgen')
    parser.add_argument("--lang", default='CoSQA')
    parser.add_argument("--hard_negative", action='store_true')
    parser.add_argument("--retrievers", nargs="+", default=["bm25", "msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3", "./models/stage1_SQL"])
    parser.add_argument("--train_type", choices=choices, type=enum_type, default='gen')
    parser.add_argument("--random_init", type=bool, default=False)
    args = parser.parse_args()

    main(args.generated_path, args.prefix, args.output_path, args.hard_negative, args.lang, args.retrievers, args.train_type)
