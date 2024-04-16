import json
import os
import sys
sys.path.append('.')
import numpy as np
import random
import argparse
import functools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import *
import jsonlines
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)
from model import Model
import multiprocessing
cpu_cont = 16

from torch.utils.data import DataLoader, SequentialSampler, BatchSampler, Dataset


def draw(X_tsne, query_samples, colors):
    # 使用t-SNE进行特征降维
    df_X1 = pd.DataFrame(X_tsne[:,0], columns=['X1'])  # Assuming X_tsne has two columns (X1 and X2)
    df_X2 = pd.DataFrame(X_tsne[:,1], columns=['X2'])


    '''    
    sns.lmplot(x='X1',
       y='X2',
       hue='L',
       data=df_merged,
       fit_reg=False,
       legend=True,
       height=9,
       scatter_kws={"s": 200, "alpha": 0.3})
    '''
    plt.figure(figsize=(8, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c='gray', cmap='rainbow', alpha=0.7, label='Labeled')
    one_indices = [index for index, element in enumerate(colors) if element == 1]
    two_indices = [index for index, element in enumerate(colors) if element == 2]
    plt.scatter(X_tsne[one_indices, 0], X_tsne[one_indices, 1], c='blue', marker='^', label='Query', s=160)
    plt.scatter(X_tsne[two_indices, 0], X_tsne[two_indices, 1], c='red', marker='o', label='Query', s=200)
    plt.scatter(X_tsne[two_indices, 0], X_tsne[two_indices, 1], c='green', marker='x', label='Query', s=220)
    plt.axis('off')
    #plt.legend()
    #plt.xlabel('t-SNE Dimension 1')
    #plt.ylabel('t-SNE Dimension 2')
    #plt.title('Active Learning Visualization with t-SNE')
    plt.show()


def create_model(args,model,tokenizer, config=None):
    # logger.info("args.data_aug_type %s"%args.data_aug_type)
    # replace token with type
    if args.data_aug_type in ["replace_type" , "other"] and not args.only_save_the_nl_code_vec:
        special_tokens_dict = {'additional_special_tokens'}
        logger.info(" new token %s"%(str(special_tokens_dict)))
        #num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

    if (args.loaded_model_filename) and ("pytorch_model.bin" in args.loaded_model_filename):
        logger.info("reload pytorch model from {}".format(args.loaded_model_filename))
        model.load_state_dict(torch.load(args.loaded_model_filename),strict=False)
        # model.from_pretrain
    if args.model_type ==  "base" :
        model= Model(model)
    elif args.model_type ==  "multi-loss-cocosoda":
        model= Multi_Loss_CoCoSoDa(model,args, args.mlp)
    if (args.loaded_model_filename) and ("pytorch_model.bin" not in args.loaded_model_filename) :
        logger.info("reload model from {}".format(args.loaded_model_filename))
        model.load_state_dict(torch.load(args.loaded_model_filename))
        # model.load_state_dict(torch.load(args.loaded_model_filename,strict=False))
        # model.from_pretrained(args.loaded_model_filename)
    if (args.loaded_codebert_model_filename) :
        logger.info("reload pytorch model from {}".format(args.loaded_codebert_model_filename))
        model.load_state_dict(torch.load(args.loaded_codebert_model_filename),strict=False)
    logger.info(model.model_parameters())


    return model

# Define your own custom batch sampler
class CustomBatchSampler(BatchSampler):
    def __init__(self, data_source, batch_size, sampler, drop_last=False):
        super().__init__(data_source, batch_size, drop_last=drop_last)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                #  position_idx,
                #  dfg_to_code,
                #  dfg_to_dfg,
                 nl_tokens,
                 nl_ids,
                 url,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        # self.position_idx=position_idx
        # self.dfg_to_code=dfg_to_code
        # self.dfg_to_dfg=dfg_to_dfg
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url=url


def convert_examples_to_features_unixcoder(js,tokenizer,args):
    """convert examples to token ids"""
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length

    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-4]
    nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length
    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids,js['url'] if "url" in js else js["retrieval_idx"])

class TextDataset_unixcoder(Dataset):
    def __init__(self, tokenizer, args, file_path=None, pooler=None):
        print(file_path)
        self.examples = []
        data = []
        n_debug_samples = 100
        with open(file_path) as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
                    if args.debug  and len(data) >= n_debug_samples:
                            break
            elif "codebase"in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
                    if  args.debug  and len(data) >= n_debug_samples:
                            break
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js)
                    if args.debug and len(data) >= n_debug_samples:
                            break
        # if "test" in file_path:
        #     data = data[-200:]
        for js in data:
            self.examples.append(convert_examples_to_features_unixcoder(js,tokenizer,args))

        if "train" in file_path:
            # self.examples = self.examples[:128]
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids))


def main(args, file_name):
    pool = multiprocessing.Pool(cpu_cont)
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaModel.from_pretrained(args.model_name_or_path)
    model = create_model(args, model, tokenizer, config)

    dataset_class = TextDataset_unixcoder
    # 指定要获取的行号
    target_line = 10  # 例如，获取第4行数据

    query_dataset = dataset_class(tokenizer, args, file_name, pool)
    query_dataset.examples = [query_dataset.examples[target_line]]
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, num_workers=4, batch_size=20)

    code_dataset = dataset_class(tokenizer, args, file_name, pool)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, num_workers=4, batch_size=20)

    train_examples1 = []
    fpath_hard_negatives = os.path.join(args.output_dir, args.lang, 'train', 'hard-negatives.jsonl')
    retrievers = ['msmarco-MiniLM-L-6-v3'] # , '',  'bm25' ,'msmarco-distilbert-base-v3','msmarco-MiniLM-L-6-v3'
    num_samples = 10
    random.seed(1024)
    with open(fpath_hard_negatives, 'r') as file:

        # 循环遍历文件的每一行
        for line_num, item in enumerate(jsonlines.Reader(file)):
            if line_num == target_line:
                negs = []
                for retr in retrievers:
                    negs.extend(item['neg'][retr][:15])

                random_samples = random.sample(negs, num_samples)
                break
    colors = [2, 3]
    for sample in code_dataset.examples[1:]:
        print(sample.url)
        if sample.url in random_samples:
            colors.append(1)
        else:
            colors.append(0)

    model.to(args.device)

    # Eval!
    logger.info("***** Running evaluation on %s *****"%args.lang)
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))


    model.eval()
    model_eval = model.module if hasattr(model,'module') else model
    code_vecs=[]
    nl_vecs=[]
    for batch in query_dataloader:
        nl_inputs = batch[-1].to(args.device)
        with torch.no_grad():
            if args.model_type ==  "base" :
                nl_vec = model(nl_inputs=nl_inputs)

            elif args.model_type in  ["cocosoda" ,"no_aug_cocosoda", "multi-loss-cocosoda"]:
                outputs = model_eval.nl_encoder_q(nl_inputs, attention_mask=nl_inputs.ne(1))
                if args.agg_way == "avg":
                    outputs = outputs [0]
                    nl_vec = (outputs*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None] # None作为ndarray或tensor的索引作用是增加维度，
                elif args.agg_way == "cls_pooler":
                    nl_vec =outputs [1]
                elif args.agg_way == "avg_cls_pooler":
                     nl_vec =outputs [1] +  (outputs[0]*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]
                nl_vec  = torch.nn.functional.normalize( nl_vec, p=2, dim=1)


            nl_vecs.append(nl_vec.tolist())

    for batch in code_dataloader:
        with torch.no_grad():
            code_inputs = batch[0].to(args.device)
            if args.model_type ==  "base" :
                code_vec = model(code_inputs=code_inputs)
            elif args.model_type in  ["cocosoda" ,"no_aug_cocosoda", "multi-loss-cocosoda"]:
                # code_vec =  model_eval.code_encoder_q(code_inputs, attention_mask=code_inputs.ne(1))[1]
                outputs = model_eval.code_encoder_q(code_inputs, attention_mask=code_inputs.ne(1))
                if args.agg_way == "avg":
                    outputs = outputs [0]
                    code_vec  = (outputs*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None] # None作为ndarray或tensor的索引作用是增加维度，
                elif args.agg_way == "cls_pooler":
                    code_vec=outputs [1]
                elif args.agg_way == "avg_cls_pooler":
                     code_vec=outputs [1] +  (outputs[0]*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]
                code_vec  = torch.nn.functional.normalize(code_vec, p=2, dim=1)


            code_vecs.extend(code_vec.tolist())


    features = nl_vecs[0] + code_vecs

    tsne = TSNE(n_components=2, random_state=11)
    X_tsne = tsne.fit_transform(np.array(features))
    #cluster_number = args.k
    #clf = KMeans(n_clusters=cluster_number, init='k-means++')
    #clf = DBSCAN(eps=0.6, min_samples=3)
    #train_label = clf.fit_predict(features)
    #cluster_centers = clf.cluster_centers_



    draw(X_tsne, None, colors)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize extracted features')
    parser.add_argument('--feature_path', default='features/CIFAR10_train.npy', type=str,
                        help='path of saved features')
    parser.add_argument('--output_dir', default='../Dataset/', type=str, help='dir to save the visualization')
    parser.add_argument('--filename', default='aft.json', type=str, help='filename of the visualization')
    parser.add_argument('--temperature', default=0.07, type=float, help='temperature for softmax')
    parser.add_argument('--threshold', default=0.0001, type=float, help='convergence threshold')
    parser.add_argument('--max_iter', default=300, type=int, help='max iterations')
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--percent', default=1, type=float, help='sample percent')
    parser.add_argument('--init', default='fps', type=str, choices=['random', 'fps'])
    parser.add_argument('--distance', default='euclidean', type=str, help='euclidean or cosine')
    parser.add_argument('--scheduler', default='none', type=str, help='scheduler')
    parser.add_argument('--balance', default=1.0, type=float, help='balance ratio')
    parser.add_argument("--lang", default='SQL', type=str, help="language to summarize")
    parser.add_argument("--add_task_prefix", default=False, action='store_true',
                        help="Whether to add task prefix for T5 and codeT5")
    parser.add_argument("--add_lang_ids", default=False, action='store_true',
                        help="Whether to add language prefix for T5 and codeT5")
    parser.add_argument("--k", default=5, type=int,
                        help="kmeans k")
    parser.add_argument("--files", default=['ids_aft.json', 'ids_km.json', 'ids_rc.json', 'ids_rd.json', 'ids_train_manual_good.json', 'ids_train_manual_bad.json'], type=str, nargs='+')
    parser.add_argument("--config_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--model_name_or_path", default="/data1/fgd/workplace/models/CoCoSoDa", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument('--data_aug_type',default="replace_type",choices=["replace_type", "random_mask" ,"other"], help="the ways of soda",required=False)
    parser.add_argument('--only_save_the_nl_code_vec', action='store_true', help='print_align_unif_loss', required=False)
    parser.add_argument("--loaded_model_filename", type=str, required=False,
                        help="loaded_model_filename")
    parser.add_argument('--model_type',default="base",choices=["base", "cocosoda","multi-loss-cocosoda","no_aug_cocosoda"], help="base is codebert/graphcoder/unixcoder",required=False)
    parser.add_argument("--loaded_codebert_model_filename", type=str, required=False,
                        help="loaded_model_filename")
    parser.add_argument('--debug', action='store_true', help='debug mode', required=False)
    parser.add_argument("--code_length", default=100, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--nl_length", default=50, type=int,
                        help="Optional NL input sequence length after tokenization.")
    args = parser.parse_args()

    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    train_filename = os.path.join(args.output_dir, args.lang, 'train.jsonl')
    print(train_filename)
    main(args, train_filename)


