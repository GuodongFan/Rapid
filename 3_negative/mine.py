import json
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
import torch
from easy_elasticsearch import ElasticSearchBM25
import tqdm
import numpy as np
import os
import logging
import argparse
import time
import edit_distance

ROOT_PATH = './Dataset/'
logger = logging.getLogger(__name__)

rel = 'rel'
gen = 'gen'
all = 'all'
choices = [rel, gen, all]


filter_false = True
threshold = 0.95

# Define a function to validate the input value
def enum_type(val):
    if val not in choices:
        raise argparse.ArgumentTypeError(f"{val} is not a valid choice.")
    return val

def sim_jaccard(s1, s2):
    """jaccard相似度"""
    s1, s2 = set(s1), set(s2)
    ret1 = s1.intersection(s2)  # 交集
    ret2 = s1.union(s2)  # 并集
    sim = 1.0 * len(ret1) / len(ret2)
    return sim

def sim_edit(s1, s2):
    sm = edit_distance.SequenceMatcher(a=s1, b=s2)
    return (len(s1)+len(s2) - sm.distance())/(len(s1)+len(s2))

class NegativeMiner(object):
    def __init__(
        self,
        lang,
        prefix,
        retrievers=["msmarco-distilbert-base-v3"], # , "msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3" "bm25"
        retriever_score_functions=["cos_sim"], # , "cos_sim", "cos_sim" "none"
        train_type: bool = False,
    ):  # , "cos_sim", "cos_sim" "none"
        nneg = 32
        print('************************')
        train_type = train_type
        generated_path = '{}{}/'.format(ROOT_PATH, lang)

        train_path = "{}{}".format(generated_path, 'train')
        print('Train Data path: {}'.format(train_path))
        print('Type: {}\nRetrievers: {}'.format(train_type, retrievers))
        print('Gen Data path: {}'.format(generated_path))

        if train_type == all:
            print('Synthetic and Real')
            self.corpus, self.gen_queries, self.gen_qrels = GenericDataLoader(
                train_path
            ).load(split="train")

            corpus_, gen_queries_, gen_qrels_ = GenericDataLoader(
                generated_path, prefix=prefix
            ).load(split="train")
            self.gen_qrels.update(gen_qrels_)
            self.gen_queries.update(gen_queries_)
            self.output_path = os.path.join(generated_path, "hard-negatives-all.jsonl")
        elif train_type == rel:
            print('Real')
            self.corpus, self.gen_queries, self.gen_qrels = GenericDataLoader(
                train_path
            ).load(split="train")
            self.output_path = os.path.join(train_path, "hard-negatives.jsonl")
        else:
            print('Synthetic')
            self.corpus, self.gen_queries, self.gen_qrels = GenericDataLoader(
                generated_path, prefix=prefix
            ).load(split="train")
            self.output_path = os.path.join(generated_path, "hard-negatives.jsonl")
        print('Output Path: {}'.format(self.output_path))
        print('************************')
        #self.output_path = os.path.join(generated_path, "hard-negatives.jsonl")
        self.retrievers = retrievers
        self.retriever_score_functions = retriever_score_functions
        if "bm25" in retrievers:
            assert (
                nneg <= 10000
            ), "Only `negatives_per_query` <= 10000 is acceptable by Elasticsearch-BM25"
            assert retriever_score_functions[retrievers.index("bm25")] == "none"

        assert set(retriever_score_functions).issubset({"none", "dot", "cos_sim"})

        self.nneg = nneg
        if nneg > len(self.corpus):
            logger.warning(
                "`negatives_per_query` > corpus size. Please use a smaller `negatives_per_query`"
            )
            self.nneg = len(self.corpus)

        #self.cross_encoder = CrossEncoder('./output/output_2stage')

    def _get_doc(self, did):
        return " ".join([self.corpus[did]["title"], self.corpus[did]["text"]])

    def _mine_sbert(self, model_name, score_function):
        logger.info(f"Mining with {model_name}")
        assert score_function in ["dot", "cos_sim"]
        normalize_embeddings = False
        if score_function == "cos_sim":
            normalize_embeddings = True

        result = {}
        sbert = SentenceTransformer(model_name)
        docs = list(map(self._get_doc, self.corpus.keys()))
        dids = np.array(list(self.corpus.keys()))
        doc_embs = sbert.encode(
            docs,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=False,
            convert_to_tensor=True,
            normalize_embeddings=normalize_embeddings,
        )
        qids = list(self.gen_qrels.keys())
        queries = list(map(lambda qid: self.gen_queries[qid], qids))
        for start in tqdm.trange(0, len(queries), 128):
            qid_batch = qids[start : start + 128]
            qemb_batch = sbert.encode(
                queries[start : start + 128],
                show_progress_bar=False,
                convert_to_numpy=False,
                convert_to_tensor=True,
                normalize_embeddings=normalize_embeddings,
            )
            score_mtrx = torch.matmul(qemb_batch, doc_embs.t())  # (qsize, dsize)
            _, indices_topk = score_mtrx.topk(k=self.nneg, dim=-1)
            indices_topk = indices_topk.tolist()
            for qid, neg_dids in zip(qid_batch, indices_topk):
                neg_dids = dids[neg_dids].tolist()

                for pos_did in self.gen_qrels[qid]:
                    if pos_did in neg_dids:
                        neg_dids.remove(pos_did)

                for pos_did in self.gen_qrels[qid]:
                    if filter_false:
                        self.code_clone_remove(neg_dids, pos_did)

                result[qid] = neg_dids
        return result

    def code_clone_remove(self, neg_dids, pos_did):
        remove_list = []
        for neg_did in neg_dids:
            code1 = self.corpus.get(neg_did)
            code2 = self.corpus.get(pos_did)
            # scores = self.cross_encoder.predict(list(zip(self.gen_queries[qid], code2)) + list(zip(self.gen_queries[qid], code1)))
            distance = sim_jaccard(code1.get('text').split(), code2.get('text').split())
            if distance > threshold:
                remove_list.append(neg_did)
        for remove_neg in remove_list:
            print('remove')
            neg_dids.remove(remove_neg)

    def _mine_bm25(self):
        logger.info(f"Mining with bm25")
        result = {}
        docs = list(map(self._get_doc, self.corpus.keys()))
        dids = np.array(list(self.corpus.keys()))
        pool = dict(zip(dids, docs))
        bm25 = ElasticSearchBM25(
            pool,
            port_http="9222",
            port_tcp="9333",
            service_type="executable",
            index_name=f"one_trial{int(time.time() * 1000000)}",
        )
        for qid, pos_dids in tqdm.tqdm(self.gen_qrels.items()):
            query = self.gen_queries[qid]
            rank = bm25.query(query, topk=self.nneg)  # topk should be <= 10000
            neg_dids = list(rank.keys())

            for pos_did in self.gen_qrels[qid]:
                if pos_did in neg_dids:
                    neg_dids.remove(pos_did)

            for pos_did in self.gen_qrels[qid]:
                if filter_false:
                    self.code_clone_remove(neg_dids, pos_did)

            result[qid] = neg_dids
        return result

    def run(self):
        hard_negatives = {}
        for retriever, score_function in zip(
            self.retrievers, self.retriever_score_functions
        ):
            if retriever == "bm25":
                hard_negatives[retriever] = self._mine_bm25()
            else:
                hard_negatives[retriever] = self._mine_sbert(retriever, score_function)

        logger.info("Combining all the data")
        result_jsonl = []
        for qid, pos_dids in tqdm.tqdm(self.gen_qrels.items()):
            line = {
                "qid": qid,
                "pos": list(pos_dids.keys()),
                "neg": {k: v[qid] for k, v in hard_negatives.items()},
            }
            result_jsonl.append(line)

        logger.info(f"Saving data to {self.output_path}")
        with open(self.output_path, "w") as f:
            for line in result_jsonl:
                f.write(json.dumps(line) + "\n")
        logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang")
    parser.add_argument("--prefix", default="qgen")
    parser.add_argument("--retrievers", nargs="+", default=["bm25", "msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3", "/data1/fgd/workplace/models/unixcoder-base"]) # "bm25", "msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"
    parser.add_argument("--retriever_score_functions", nargs="+", default=["none", "cos_sim", "cos_sim", "cos_sim"]) # "none", "cos_sim", "cos_sim"
    parser.add_argument("--train_type", choices=choices, type=enum_type, default='gen')
    args = parser.parse_args()

    miner = NegativeMiner(args.lang, args.prefix, args.retrievers, args.retriever_score_functions, args.train_type)
    miner.run()
