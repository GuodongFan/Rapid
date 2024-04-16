import os
import json
import numpy as np
import random
import re
import hashlib

langs = ['SQL', 'Solidity', 'CoSQA']
file_types = ['train', 'valid', 'test']
# 格式
# "_id" "text", "title", "metadata"

# code_tokens docstring_tokens
def format_test():

    for lang in langs:

        for file_type in file_types:
            data = []
            FILE_DATA_DIR = './Dataset/{}/{}.txt'.format(lang, file_type)
            with open(FILE_DATA_DIR, "r", encoding="utf-8") as f:
                datas = f.readlines()

            l = len(datas)
            for idx in range(0, l):
                #if idx == 1000:
                #    break
                doc_token = datas[idx].split("<CODESPLIT>")
                if doc_token[0] == '0':
                    continue

                code = doc_token[4].split()
                query = doc_token[3].split()

                url_text = doc_token[4]
                url_text = str(url_text).encode('utf-8')
                md5hash = hashlib.md5(url_text)
                md5 = md5hash.hexdigest()

                data.append(json.dumps({'code_tokens':code, 'docstring_tokens':query, 'url': md5})+'\n')

            print("lang {} type {} len {}".format( lang, file_type, len(data)))
            with open('./Dataset/{}/{}.jsonl'.format(lang, file_type), 'w') as f:
                f.writelines(data)



if __name__ == '__main__':
    format_test()