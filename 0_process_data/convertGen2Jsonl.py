import random

from beir.datasets.data_loader import GenericDataLoader
import argparse
import json
import hashlib
import os
from dataset import HardNegativeDataset
import jsonlines


def convert(generated_path, prefix, lang, file_type='train'):

    train_examples = []
    train_examples1 = []

    retrivers = ['bm25', 'msmarco-distilbert-base-v3', 'msmarco-MiniLM-L-6-v3']
    corpus, gen_queries, gen_qrels = GenericDataLoader(
        generated_path, prefix=prefix
    ).load(split="train")


    swapped_dict = {}
    for key, val in gen_qrels.items():

        for i_key, i_val in val.items():
            if swapped_dict.get(i_key) == None:
                swapped_dict[i_key] = [key]
            else:
                swapped_dict[i_key].append(key)

    fpath_hard_negatives = os.path.join(generated_path,
                                        'hard-negatives.jsonl')
    hard_negative_dataset = HardNegativeDataset(
        fpath_hard_negatives, gen_queries, corpus, retrivers
    )

    hash_set =set()
    for qid, cidic in gen_qrels.items():
        query = gen_queries.get(qid)
        for cid, _ in cidic.items():

            code = corpus.get(cid)['text']

            url_text = code
            url_text = str(url_text).encode('utf-8')
            md5hash = hashlib.md5(url_text)
            md5 = md5hash.hexdigest()
            if md5 in hash_set:
                continue
            hash_set.add(md5)
            train_examples.append(json.dumps({'code_tokens':code.split(), 'docstring_tokens':query.split(), 'url': md5})+'\n')


            item = hard_negative_dataset.get_queryid(qid)
            example = hard_negative_dataset.__getitem__(item)
            query = example.texts[0]
            code = example.texts[1]['text']
            url_text = code + query
            url_text = str(url_text).encode('utf-8')
            md5hash = hashlib.md5(url_text)
            md5 = md5hash.hexdigest()
            train_examples1.append(json.dumps({'code_tokens': code.split(), 'docstring_tokens': query.split(), 'url': md5}) + '\n')

    print("lang {} type {} len {}".format( lang, file_type, len(train_examples)))
    with open('./Dataset/{}/{}.jsonl'.format(lang, file_type), 'w') as f:
        f.writelines(train_examples)
    print("lang {} type {} len {}".format( lang, file_type, len(train_examples1)))
    #with open('./Dataset/{}/{}_.jsonl'.format(lang, file_type), 'w') as f:
    #    f.writelines(train_examples1)


    train_examples1 = []
    fpath_hard_negatives = os.path.join(generated_path, 'hard-negatives.jsonl')
    retrievers = ['bm25', 'msmarco-distilbert-base-v3', 'msmarco-MiniLM-L-6-v3']
    num_samples = 1
    with open(fpath_hard_negatives, 'r') as file:
        for item in jsonlines.Reader(file):
            negs = []
            for retr in retrievers:
                negs.extend(item['neg'][retr][:5])

            random_samples = random.sample(negs, num_samples)
            for sample in random_samples:
                neg_code = corpus.get(sample)['text']
                neg_query_id = random.choice(swapped_dict.get(sample))
                neg_query = gen_queries.get(neg_query_id)
                train_examples1.append(json.dumps({'code_tokens': neg_code.split(), 'docstring_tokens': neg_query.split(), 'url': md5}) + '\n')

    print("lang {} type {} len {}".format( lang, file_type, len(train_examples1)))
    with open('./Dataset/{}/{}_.jsonl'.format(lang, file_type), 'w') as f:
        f.writelines(train_examples1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_path", default='/data1/fgd/workplace/ZDAN/Dataset/')
    parser.add_argument("--output_path", default='./models/')
    parser.add_argument("--prefix", default='')
    parser.add_argument("--lang", default='SQL')
    parser.add_argument("--hard_negative", action='store_true')
    parser.add_argument("--retrievers", nargs="+", default=["bm25", "msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"])
    parser.add_argument("--train_type", default='')
    parser.add_argument("--random_init", type=bool, default=False)
    args = parser.parse_args()
    generated_path = "{}{}/train/".format(args.generated_path, args.lang)
    print(generated_path)
    convert(generated_path, args.prefix, args.lang)
