import json
from typing import Dict
from torch.utils.data import Dataset
from sentence_transformers.readers.InputExample import InputExample
import random
import linecache
import logging

logger = logging.getLogger(__name__)


def concat_title_and_body(did: str, corpus: Dict[str, Dict[str, str]], sep: str):
    assert type(did) == str
    document = []
    title = corpus[did]["title"].strip()
    body = corpus[did]["text"].strip()
    if len(title):
        document.append(title)
    if len(body):
        document.append(body)
    return sep.join(document)


class HardNegativeDataset(Dataset):
    def __init__(self, jsonl_path, queries, corpus, ranks, sep=" "):
        self.jsonl_path = jsonl_path
        self.queries = queries
        self.corpus = corpus
        self.sep = sep
        self.none_indices = set()
        self.nqueries = len(linecache.getlines(jsonl_path))
        self.ranks = ranks

        self.query_id_dic = {}
        for idx, query in enumerate(linecache.getlines(jsonl_path)):
            self.query_id_dic.update({json.loads(query)['qid']: idx})
        self.pos_id_dic = {}
        for idx, query in enumerate(linecache.getlines(jsonl_path)):
            for pos in json.loads(query)['pos']:
                self.pos_id_dic.update({pos: idx})

    def get_queryid(self, query):
        #query = random.choice(list(range(self.nqueries)))
        #return query
        return self.query_id_dic.get(query)

    def __getitem__(self, item):
        shift = 0
        while True:
            index = item + shift + 1
            shift += 1
            if index in self.none_indices:
                continue
            json_line = linecache.getline(self.jsonl_path, index)
            try:
                query_dict = json.loads(json_line)
            except:
                print(json_line, "###index###", index)
                item = 0
                continue
                #raise NotImplementedError
            tuple_sampled = self._sample_tuple(query_dict)
            if tuple_sampled is None:
                self.none_indices.add(index)
                logger.info(f"Invalid query at line {index-1}")
            else:
                break
        (query_id, pos_id, neg_id), (query_text, pos_text, neg_text) = tuple_sampled
        neg_idx = self.pos_id_dic.get(neg_id)
        if neg_idx == None:
            print(neg_id)
        json_line = linecache.getline(self.jsonl_path, neg_idx+1)
        query_dict = json.loads(json_line)
        query_id = query_dict["qid"]
        query_text = self.queries[query_dict["qid"]]
        return InputExample(
            guid=[query_id, neg_id],
            texts=[query_text, {'text':neg_text, 'title':''}],
            label=0,
        )

    def __len__(self):
        return self.nqueries

    def _sample_tuple(self, query_dict):
        # Get the positive passage ids
        pos_pids = query_dict["pos"]
        # scores = {item['pid']: item['ce-score'] for item in query_dict['pos']}

        # Get the hard negatives
        neg_pids = set()
        sys_list = []
        for system_name, system_negs in query_dict["neg"].items():
            if system_name not in self.ranks:
                #print('continue {}'.format(system_name))
                continue
            sys_list.append(system_negs)


        for idx_list in range(32):
            for idx_sys in range(len(sys_list)):
                if len(sys_list[idx_sys]) <= idx_list:
                    continue
                if len(neg_pids) >= 32:
                    break
                neg_pids.add(sys_list[idx_sys][idx_list])


        if len(pos_pids) > 0 and len(neg_pids) > 0:
            query_text = self.queries[query_dict["qid"]]

            pos_pid = random.choice(pos_pids)
            pos_text = concat_title_and_body(pos_pid, self.corpus, self.sep)

            neg_pid = random.choice(list(neg_pids))
            neg_text = concat_title_and_body(neg_pid, self.corpus, self.sep)

            return (query_dict["qid"], pos_pid, neg_pid), (
                query_text,
                pos_text,
                neg_text,
            )
        else:
            return None


class GenerativePseudoLabelingDataset(Dataset):
    def __init__(self, tsv_path, queries, corpus, sep=" "):
        self.tsv_path = tsv_path
        self.queries = queries
        self.corpus = corpus
        self.sep = sep
        self.ntuples = len(linecache.getlines(tsv_path))

    def __getitem__(self, item):
        index = item + 1
        tsv_line = linecache.getline(self.tsv_path, index)
        qid, pos_pid, neg_pid, label = tsv_line.strip().split("\t")
        query_text = self.queries[qid]
        pos_text = concat_title_and_body(pos_pid, self.corpus, self.sep)
        neg_text = concat_title_and_body(neg_pid, self.corpus, self.sep)
        label = float(label)  # CE margin between (query, pos) and (query, neg)

        return InputExample(texts=[query_text, pos_text, neg_text], label=label)

    def __len__(self):
        return self.ntuples

def hard_negative_collate_fn(batch):
    query_id, pos_id, neg_id = zip(*[example.guid for example in batch])
    query, pos, neg = zip(*[example.texts for example in batch])
    return (query_id, pos_id, neg_id), (query, pos, neg)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from beir.datasets.data_loader import GenericDataLoader
    import os
    import tqdm

    generated_path = './Dataset/SQL/'
    corpus, gen_queries, gen_qrels = GenericDataLoader(
        generated_path, prefix='qgen'
    ).load(split="train")
    generated_path = './Dataset/SQL/'
    fpath_hard_negatives = os.path.join(generated_path, "hard-negatives.jsonl")
    hard_negative_dataset = HardNegativeDataset(
        fpath_hard_negatives, gen_queries, corpus
    )
    hard_negative_dataloader = DataLoader(
        hard_negative_dataset, shuffle=False, batch_size=10, drop_last=True
    )

    hard_negative_dataloader.collate_fn = hard_negative_collate_fn
    hard_negative_iterator = iter(hard_negative_dataloader)
    logger.info("Begin pseudo labeling")
    for _ in tqdm.trange(10):
        try:
            batch = next(hard_negative_iterator)
        except StopIteration:
            hard_negative_iterator = iter(hard_negative_dataloader)
            batch = next(hard_negative_iterator)

        print('')
