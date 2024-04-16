import json
from tqdm import tqdm
import hashlib
import random
import os
from sklearn.model_selection import train_test_split

file_dir = './Dataset/review2code'

files_path = ['termux', 'k9mail', 'cgeo', 'anki', 'AntennaPod']

SAMPLE_NUM = 5

# md5加密
def md5_string(in_str):
    md5 = hashlib.md5()
    md5.update(in_str.encode("utf8"))
    result = md5.hexdigest()
    return result

def add_neg_sample_reivews(data_list):
    train_data_list = []
    id2method = dict()
    review2idlist= dict()

    for idx, dic in tqdm(enumerate(data_list)):
        dic['review_id'] = idx


        review_raw = dic['review_raw']
        method_path = dic['method_path']
        method_name = dic['method_name']
        method_content = dic['method_content']
        method_id = method_path + '#' + method_name

        m2 = hashlib.md5()
        m2.update(review_raw.encode(encoding='utf-8'))
        review_id = m2.hexdigest()

        if review_id not in review2idlist:
            review2idlist[review_id] = {
                'review_raw':review_raw,
                'method_list':set([method_id])
            }
        else:
            review2idlist[review_id]['method_list'].add(method_id)

        if method_id not in id2method:
            id2method[method_id]={
                'method_path':method_path,
                'method_name':method_name,
                'method_content':method_content
            }


    for key, val in review2idlist.items():
        neg_num = 0
        pos_num = 0
        for key_, val_ in review2idlist.items():
            if key != key_:
                score = len(val['method_list'] & val_['method_list'])/len(val['method_list'] | val_['method_list'])
                review1 = val['review_raw']
                review2 = val_['review_raw']
                if score > 0.6 and pos_num < SAMPLE_NUM:
                    pos_num = pos_num + 1
                    pos_data = {

                        'review_raw': review1,
                        'review_raw2': review2,
                        'label': score
                    }
                    train_data_list.append(pos_data)
                    continue
                if score < 0.01 and neg_num < SAMPLE_NUM:
                    neg_num = neg_num + 1
                    neg_data = {

                        'review_raw': review1,
                        'review_raw2': review2,
                        'label': 0
                    }
                    train_data_list.append(neg_data)
                    continue

    return train_data_list


def add_neg_sample(data_list):
    id2method = dict()
    review2idlist= dict()

    for idx, dic in tqdm(enumerate(data_list)):
        dic['review_id'] = idx


        review_raw = dic['review_raw']
        method_path = dic['method_path']
        method_name = dic['method_name']
        method_content = dic['method_content']
        method_id = method_path + '#' + method_name

        review_id = md5_string(review_raw)

        if review_id not in review2idlist:
            review2idlist[review_id] = {
                'review_raw':review_raw,
                'method_list':set([method_id])
            }
        else:
            review2idlist[review_id]['method_list'].add(method_id)

        if method_id not in id2method:
            id2method[method_id]={
                'method_path':method_path,
                'method_name':method_name,
                'method_content':method_content
            }


    # 接下来是构造样本
    all_id_list = list(id2method.keys())
    new_data_list = []
    for review_id in tqdm(review2idlist):
        review_raw = review2idlist[review_id]['review_raw']
        method_list = review2idlist[review_id]['method_list']

        # 确保每个user review对应的方法唯一
        id_set = set(method_list)
        for method_id in method_list:
            # 首先构造正样本
            pos_data = {
                'review_id':review_id,
                'review_raw':review_raw,
                'method_path':id2method[method_id]['method_path'],
                'method_name':id2method[method_id]['method_name'],
                'method_content':id2method[method_id]['method_content'],
                'label':1
            }

            new_data_list.append(pos_data)
            # 构造负样本
            # 随机选择一个负样本
            choose_id = random.choice(all_id_list)
            while choose_id in id_set:
                choose_id = random.choice(all_id_list)
            id_set.add(choose_id)
            neg_data = {
                'review_id':review_id,
                'review_raw':review_raw,
                'method_path':id2method[choose_id]['method_path'],
                'method_name':id2method[choose_id]['method_name'],
                'method_content':id2method[choose_id]['method_content'],
                'label':0
            }
            new_data_list.append(neg_data)
    return new_data_list


for cur_name in files_path:
    # 每个code只配一个review
    cur_file = cur_name+'/'+cur_name + '_GT.json'
    with open(os.path.join(file_dir, cur_file), 'r', encoding='utf-8') as f:
        all_list = json.load(f)

        id2method = dict()
        review2idlist = dict()

        # 格式
        # "_id" "text", "title", "metadata"
        output_list = []
        spliter = '<CODESPLIT>'

        for idx, dic in tqdm(enumerate(all_list)):

            review_raw = dic['review_raw']
            method_path = dic['method_path']
            method_name = dic['method_name']
            method_content = dic['method_content']
            method_id = method_path + '#' + method_name
            review_raw = review_raw.replace('\n', ' ')
            method_content = method_content.replace('\n', ' ')

            prompt_path = '. the path of the code is {}'.format(method_path)
            prompt_path = prompt_path.replace('/', ' ')
            method_content = method_content + prompt_path

            review_id = md5_string(review_raw)

            if review_id not in review2idlist:
                review2idlist[review_id] = {
                    'review_raw': review_raw,
                    'method_list': set([method_id])
                }
            else:
                review2idlist[review_id]['method_list'].add(method_id)

            if method_id not in id2method:
                id2method[method_id] = {
                    'method_path': method_path,
                    'method_name': method_name,
                    'method_content': method_content
                }
            # 格式
            # "_id" "text", "title", "metadata"
            content = '1'+spliter+id2method[method_id]['method_path']+spliter+id2method[method_id]['method_name']+spliter+review_raw+spliter+id2method[method_id]['method_content']+'\n'
            output_list.append(content)
        print(cur_file)
        print('review len: {}'.format(len(review2idlist)))
        print('method len: {}'.format(len(id2method)))

        # old
        train, test = train_test_split(output_list, test_size=0.3, random_state=1024)
        test, eval = train_test_split(test, test_size=0.5, random_state=1024)

        #train_data = add_neg_sample(train)
        train_path = cur_name+'/' + 'train.txt'
        eval_path = cur_name+'/' + 'valid.txt'
        test_path = cur_name+'/' + 'test.txt'
        print('train data: {} {}'.format(len(train), len(train)))
        with open(os.path.join(file_dir, train_path), 'w', encoding='utf-8') as f:
            f.writelines(train)
        print('eval data: {}'.format(len(eval)))
        with open(os.path.join(file_dir, eval_path), 'w', encoding='utf-8') as f:
            f.writelines(eval)

        print('test data: {}'.format(len(test)))
        with open(os.path.join(file_dir, test_path), 'w', encoding='utf-8') as f:
            f.writelines(test)
