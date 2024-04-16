import os
import json
import numpy as np
import random
import re

import tree_sitter
from pprint import pprint
from tree_sitter import Language, Parser


var_names = []
# Define a function to traverse the syntax tree
def traverse_identifiers(node):
    #print(node.type)
    #print(node.text)
    if node.type == 'identifier':
        var_names.append(node.text)
        #var_name_node = node
        #print(node.text)
    for child in node.children:
        traverse_identifiers(child)



import hashlib
# md5加密
def md5_string(in_str):
    md5 = hashlib.md5()
    md5.update(in_str.encode("utf8"))
    result = md5.hexdigest()
    return result


#langs = ['SQL', 'Solidity', 'CoSQA', 'review2code/termux', 'ruby']
langs = ['advtest/airflow', 'advtest/gurumate', 'advtest/scout']
file_types = ['repo_test']
# 格式
# "_id" "text", "title", "metadata"

def format_test():

    for lang in langs:
        # Load the Python parser
        if lang == 'SQL':
            lan = 'sql_bigquery'
        elif lang == 'Solidity':
            lan = 'javascript'
        else:
            lan = 'python'

        language = tree_sitter.Language('libs/build/my-languages.so', lan)
        parser = Parser()
        parser.set_language(language)

        all_examples = []
        all_queres = []
        for file_type in file_types:

            md5_set = set()

            queres = []
            examples = []

            qrels = ['query-id\tcorpus-id\tscore\n']

            FILE_DATA_DIR = './Dataset/{}/{}.jsonl'.format(lang, file_type)
            #FILE_DATA_DIR = './SQL.txt'.format(lang)
            ROOT_DIR = './Dataset/{}/'.format(lang)
            OUTPUT_DIR = ROOT_DIR + '{}/'.format(file_type)


            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)

            with open(FILE_DATA_DIR, "r", encoding="utf-8") as f:
                datas = f.readlines()

            code_dic = {}
            review_dic = {}
            l = len(datas)
            for idx in range(0, l):
                #if idx == 1000:
                #    break

                d = json.loads(datas[idx])

                '''query = d['docstring']
                code = d['code']'''
                query = d['docstring_summary']
                code = ' '.join(d['function_tokens'])
                print(idx)

                code_md5 = md5_string(code)
                review_md5 = md5_string(query)
                _id_code = 'c_' + str(idx)
                _id_query = 'q_' + str(idx)
                _id_code = code_md5
                _id_query = review_md5
                code_dic.update({_id_code: code})
                review_dic.update({_id_query: query})

                #过滤相同样本
                id_md5 = md5_string( query + '**' + code)
                if id_md5 in md5_set:
                    continue
                else:
                    md5_set.add(id_md5)

                #code = json.dumps({'_id': _id_code, 'text': doc_token[4], 'title': '', 'metadata': ''}) + '\n'
                #examples.append(code)

                # 数据结构中一个query对应一个code，但是review的时候可以对应多个，所以这里不对query去重了。
                #query = json.dumps({'_id': _id_query, 'text': doc_token[3], 'title': '', 'metadata': ''}) + '\n'
                #queres.append(query)
                qrels.append(_id_query + '\t' + _id_code + '\t1\n')


            for key, val in code_dic.items():
                code = json.dumps({'_id': key, 'text': val, 'title': '', 'metadata': ''}) + '\n'
                examples.append(code)
            for key, val in review_dic.items():
                query = json.dumps({'_id': key, 'text': val, 'title': '', 'metadata': ''}) + '\n'
                queres.append(query)

            print('code len: {}', len(examples))

            data_path = OUTPUT_DIR
            if not os.path.exists(data_path):
                os.makedirs(data_path)
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


            if file_type != 'repo_test': # 这里只取test 之前是train
                continue

            all_queres.extend(queres)
            all_examples.extend(examples)

        # 总文件
        file_path = os.path.join(ROOT_DIR, 'corpus.jsonl')
        with open(file_path, 'w+') as f:
            f.writelines(all_examples)

        #file_path = os.path.join(OUTPUT_DIR, 'queries.jsonl')
        #with open(file_path, 'w+') as f:
            #f.writelines(all_queres)
if __name__ == '__main__':
    format_test()