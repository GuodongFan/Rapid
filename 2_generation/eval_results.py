from nlgeval import NLGEval
import argparse
import json
from beir.datasets.data_loader import GenericDataLoader

nlgeval = NLGEval(no_skipthoughts=True,
                               no_glove=True)


def main(lang, ques_per_passage, prefix):

    groundtruth_file = './Dataset/{}/train/queries.jsonl'.format(lang)
    predict_file = './Dataset/{}/qgen-queries.jsonl'.format(lang)
    generated_path = './Dataset/{}/'.format(lang)
    train_path = '{}{}'.format(generated_path, 'train')

    groundtruth_list = []
    predict_list = []

    corpus, gen_queries, gen_qrels = GenericDataLoader(
        generated_path, prefix=prefix
    ).load(split="train")

    corpus, queries, qrels = GenericDataLoader(
        train_path
    ).load(split="train")

    reverse_qrels = {}
    for key, val in qrels.items():
        cid = val.popitem()[0]
        reverse_qrels.update({cid: key})

    for key,val in gen_qrels.items():
        cid = val.popitem()[0]
        qid = reverse_qrels.get(cid)
        predict_list.append(gen_queries.get(key))
        groundtruth_list.append(queries.get(qid))

    print(nlgeval.compute_metrics(hyp_list = predict_list, ref_list = [groundtruth_list]))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # outdated parameters
    parser.add_argument("--lang", default='Solidity')
    parser.add_argument("--ques_per_passage", default=3)
    parser.add_argument("--prefix", default='qgen_full')

    # print arguments
    args = parser.parse_args()

    main(args.lang, args.ques_per_passage, args.prefix)
