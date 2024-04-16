import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

train_csv = pd.read_csv('./Dataset/SO/Train.csv', nrows=300000)
#train_csv=train_csv.dropna(subset=['Title','Body', 'Tags'],how='any')
X_train, X_val_test = train_test_split(train_csv, test_size=0.02, random_state=1024)
X_test, X_val = train_test_split(X_val_test, test_size=0.4, random_state=1024)

count = 0



file_types = ['train', 'valid', 'test']
split_files = [X_train, X_val, X_test]
for type_idx, file_type in enumerate(file_types):
    queres = []
    examples = []
    query_tags = []
    qrels = ['query-id\tcorpus-id\tscore\n']

    all_tags = set()
    tag_dic = {}
    for idx, line in split_files[type_idx].iterrows():
        tags = line['Tags']
        for tag in tags.split(' '):
            tag_dic.update({tag: idx})

    for idx, line in split_files[type_idx].iterrows():

        title = line['Title']
        body = line['Body'].replace('\n', ' ')
        tags = line['Tags']

        _id_code = 'c_' + str(idx)
        _id_query = 'q_' + str(idx)

        query = json.dumps({'_id': _id_query, 'text': title, 'title': '', 'metadata': ''}) + '\n'
        code = json.dumps({'_id': _id_code, 'text': body, 'title': '', 'metadata': ''}) + '\n'
        queres.append(query)
        examples.append(code)
        qrels.append(_id_query + '\t' + _id_code + '\t1\n')

        '''
        split_tags = tags.split(' ')
        for tag in split_tags:
            _id_code = 'c_' + str(tag_dic.get(tag))
            _id_query = 'q_' + str(idx)

            query = json.dumps({'_id': _id_query, 'text': body, 'title': '', 'metadata': ''}) + '\n'
            queres.append(query)
            qrels.append(_id_query + '\t' + _id_code + '\t1\n')

            all_tags.update([tag])

    for idx, tag in enumerate(all_tags):
        _id_code = 'c_' + str(tag_dic.get(tag))
        prompt = 'this is a question with a {} tag'.format(tag)
        code = json.dumps({'_id': _id_code, 'text': prompt, 'title': '', 'metadata': ''}) + '\n'
        examples.append(code)'''

    data_path =  './Dataset/{}/'.format('SO')
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    OUTPUT_DIR = data_path + '{}/'.format(file_type)
    qrels_path = OUTPUT_DIR + '/qrels/'
    if not os.path.exists(qrels_path):
        os.makedirs(qrels_path)
    file_path = os.path.join(OUTPUT_DIR, 'corpus.jsonl')
    with open(file_path, 'w+') as f:
        f.writelines(examples)

    file_path = os.path.join(OUTPUT_DIR, 'queries.jsonl')
    with open(file_path, 'w+') as f:
        f.writelines(queres)

    file_path = os.path.join(OUTPUT_DIR, 'qrels/{}.tsv'.format(file_type))
    with open(file_path, 'w+') as f:
        f.writelines(qrels)


queres = []
examples = []
query_tags = []
qrels = ['query-id\tcorpus-id\tscore\n']

all_tags = set()
tag_dic = {}

all_tags = set()
tag_dic = {}
count_idx = 0
for idx, line in X_test.iterrows():
    tags = line['Tags']
    split_tags = tags.split(' ')
    all_tags.update(split_tags)
    for tag in split_tags:
        if tag_dic.get(tag) == None:
            tag_dic.update({tag: count_idx})
            count_idx += 1

for idx, line in X_test.iterrows():

    title = line['Title']
    body = line['Body'].replace('\n', ' ')
    tags = line['Tags']
    split_tags = tags.split(' ')

    for tag_id in split_tags[0:1]:
        _id_code = 'c_' + str(tag_dic.get(tag_id))
        _id_query = 'q_' + str(idx)

        query = json.dumps({'_id': _id_query, 'text': body, 'title': '', 'metadata': ''}) + '\n'
        queres.append(query)

        query_tag = json.dumps({'_id': '', 'text': tag_id, 'title': '', 'metadata': ''}) + '\n'
        query_tags.append(query_tag)

        prompt = 'the topic of the passage is {}'.format(tag_id)
        code = json.dumps({'_id': _id_code, 'text': prompt, 'title': '', 'metadata': ''}) + '\n'
        #examples.append(code)

        qrels.append(_id_query + '\t' + _id_code + '\t1\n')

for idx, tag in enumerate(all_tags):
    prompt = 'the topic of the passage is {}'.format(tag)
    _id_code = 'c_' + str(tag_dic.get(tag))
    code = json.dumps({'_id': _id_code, 'text': prompt, 'title': '', 'metadata': ''}) + '\n'
    examples.append(code)

data_path =  './Dataset/{}/'.format('SO')
if not os.path.exists(data_path):
    os.makedirs(data_path)

OUTPUT_DIR = data_path + '{}/'.format('test')
qrels_path = OUTPUT_DIR + '/qrels/'
if not os.path.exists(qrels_path):
    os.makedirs(qrels_path)
file_path = os.path.join(OUTPUT_DIR, 'corpus.jsonl')
with open(file_path, 'w+') as f:
    f.writelines(examples)

file_path = os.path.join(OUTPUT_DIR, 'queries.jsonl')
with open(file_path, 'w+') as f:
    f.writelines(queres)

file_path = os.path.join(OUTPUT_DIR, 'tags.jsonl')
with open(file_path, 'w+') as f:
    f.writelines(query_tags)

file_path = os.path.join(OUTPUT_DIR, 'qrels/{}.tsv'.format('test'))
with open(file_path, 'w+') as f:
    f.writelines(qrels)
