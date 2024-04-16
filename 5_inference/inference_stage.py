import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import gzip
import os
import numpy as np
import torch
from beir.datasets.data_loader import GenericDataLoader
import argparse

rel = 'rel'
gen = 'gen'
all = 'all'
choices = [rel, gen, all]

# Define a function to validate the input value
def enum_type(val):
    if val not in choices:
        raise argparse.ArgumentTypeError(f"{val} is not a valid choice.")
    return val

if not torch.cuda.is_available():
    print("Warning: No GPU found. Please add GPU to your notebook")


# answer the query
def search(query, bi_encoder, cross_encoder, corpus_embeddings, passages, passages_ids, top_k):
    stage1_list = []
    stage2_list = []

    print("Input question:", query)

    ##### Sematic Search #####
    # Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding.cuda()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    if False:
        cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
        cross_scores = cross_encoder.predict(cross_inp)

        # Sort results by the cross-encoder scores
        for idx in range(len(cross_scores)):
            hits[idx]['cross-score'] = cross_scores[idx]

    # Output of top-5 hits from bi-encoder
    print("\n-------------------------\n")
    print("Top-3 Bi-Encoder Retrieval hits")
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    for hit in hits[0:3]:
        print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))
    for hit in hits:
        stage1_list.append(passages_ids[hit['corpus_id']])

    # Output of top-5 hits from re-ranker
    if False:
        print("\n-------------------------\n")
        print("Top-3 Cross-Encoder Re-ranker hits")
        hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
        for hit in hits[0:3]:
            print("\t{:.3f}\t{}".format(hit['cross-score'], passages[hit['corpus_id']].replace("\n", " ")))

        for hit in hits:
            stage2_list.append(hit['corpus_id'])

    return stage1_list, stage2_list

def calculate_metric(sort_id_list, query_num, gen_qrels):
    top1_hitting = 0
    top3_hitting = 0
    top5_hitting = 0
    top10_hitting = 0
    top20_hitting = 0
    ranks = []
    result_list = []

    for idx, sort_id in enumerate(sort_id_list):

        real_code_list = []
        real_code_list.extend(list(list(gen_qrels.values())[idx]))
        code_information_list = []
        code_information_list.extend(sort_id)
        # 计算MRR
        rank = 0
        find = False
        for id in sort_id[:1000]:
            if find is False:
                rank += 1
            if id in real_code_list:
                find = True
                ranks.append(1 / rank)
                break
        if not find:
            ranks.append(0)

        predict_code_path_list = []

        if len(set(real_code_list) & set(code_information_list[:1])) != 0:
            top1_hitting += 1
        if len(set(real_code_list) & set(code_information_list[:3])) != 0:
            top3_hitting += 1
        if len(set(real_code_list) & set(code_information_list[:5])) != 0:
            top5_hitting += 1
        if len(set(real_code_list) & set(code_information_list[:10])) != 0:
            top10_hitting += 1
        if len(set(real_code_list) & set(code_information_list[:20])) != 0:
            top20_hitting += 1
    print('MRR:', str(np.mean(ranks)))
    print('top1_hitting:', str(top1_hitting / query_num))
    print('top3_hitting:', str(top3_hitting / query_num))
    print('top5_hitting:', str(top5_hitting / query_num))
    print('top10_hitting:', str(top10_hitting / query_num))
    print('top20_hitting:', str(top20_hitting / query_num))


def main(generated_path, output_path, hard_negative, lang, retrievers, train_type, model_path):

    print('************************')
    print('lang {} \nhard {} \nretrievers {} \ntrain type {}'.format(lang, hard_negative, retrievers, train_type))

    generated_path = "{}{}".format(generated_path, lang)
    print('Generated Path: {}'.format(generated_path))

    test_path = "{}/test/".format(generated_path)
    print('Test Path: {}'.format(test_path))



    if hard_negative:
        stage1_path = "{}stage1_{}_{}".format(output_path, lang, "hard")
        stage2_path = "{}stage2_{}".format(output_path, lang)
        for retriever in retrievers:
            stage1_path = "{}_{}".format(stage1_path, retriever)
    else:
        stage1_path = "{}stage1_{}".format(output_path, lang)
        stage2_path = "{}stage2_{}".format(output_path, lang)

    if train_type == rel:
        stage1_path = "{}_rel".format(stage1_path)
        stage2_path = "{}_rel".format(stage2_path)
    elif train_type == all:
        stage1_path = "{}_all".format(stage1_path)
        stage2_path = "{}_all".format(stage2_path)

    if model_path != None:
        stage1_path = model_path

    print('Model 1 Path: {}'.format(stage1_path))

    print('************************')
    config = {'stage1': stage1_path,
              'stage2': stage2_path}

    if False:
        stage1_path = './models/pre-trained/msmarco-MiniLM-L-6-v3/'

    #We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
    bi_encoder = SentenceTransformer(stage1_path)
    bi_encoder.max_seq_length = 256     #Truncate long passages to 256 tokens
    top_k = 20                          #Number of passages we want to retrieve with the bi-encoder

    #The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
    cross_encoder = None
    if False:
        cross_encoder = CrossEncoder(stage2_path)

    corpus, gen_queries, gen_qrels = GenericDataLoader(
        test_path, prefix=''
    ).load(split="test")

    passages = []
    passages_ids = []
    for cid, code_dic in corpus.items():
        code = code_dic.get('text')

        #Add all paragraphs
        #passages.extend(data['paragraphs'])

        #Only add the first paragraph
        passages.append(code)
        passages_ids.append(cid)


    queries = []
    queries_ids = []
    for qid, query in gen_queries.items():
        queries.append(query)
        queries_ids.append(qid)

    print("Passages:", len(passages))

    # We encode all passages into our vector space. This takes about 5 minutes (depends on your GPU speed)
    corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True)


    # This function will search all wikipedia articles for passages that

    query_num = 0

    # 存储分数
    work_idx = 0
    work_total = len(queries)
    sort_id_list1 = []
    sort_id_list = []
    for query in queries:
        if work_idx % 10 == 0:
            print('process ', str(work_idx), ' total ', str(work_total))
        work_idx += 1
        stage1_list, stage2_list = search(query, bi_encoder, cross_encoder, corpus_embeddings, passages, passages_ids, top_k)
        sort_id_list.append(stage2_list)
        sort_id_list1.append(stage1_list)
        query_num += 1

    calculate_metric(sort_id_list1, query_num, gen_qrels)
    calculate_metric(sort_id_list, query_num, gen_qrels)


if __name__ == "__main__":

    # --generated_path Dataset/SQL/groundtruth --output_path ./output/output_1stage_supervised
    # --generated_path Dataset/SQL/ --prefix qgen --output_path ./output/output_1stage
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_path", default='./Dataset/')
    parser.add_argument("--output_path", default='./models/')
    parser.add_argument("--lang", default='SQL')
    parser.add_argument("--supervised", action='store_true', default=False)
    parser.add_argument("--hard_negative", action='store_true')
    parser.add_argument("--retrievers", nargs="+", default=["bm25"])
    parser.add_argument("--train_type", choices=choices, type=enum_type, default='gen')
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    main(args.generated_path, args.output_path, args.hard_negative, args.lang, args.retrievers, args.train_type, args.model)
