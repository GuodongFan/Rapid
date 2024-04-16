import os
import json
import numpy as np
import random
import re
from sklearn.model_selection import train_test_split

langs = ['SQL', 'Solidity', 'CoSQA']
file_types = ['train', 'valid', 'test']
# 格式
# "_id" "text", "title", "metadata"
#1<CODESPLIT>URL<CODESPLIT>func_name<CODESPLIT>Replace the owner with a new owner .<CODESPLIT>contract  c29479{ /** *  @dev  Replace  the  owner  with  a  new  owner *  @dev  Transaction  has  to  be  sent  by  wallet *  @param  owner  The  address  of  owner  to  be  replaced *  @param  newOwner  The  address  of  new  owner */ function  replaceOwner(address  owner,  address  newOwner) public onlyWallet onlyOwnerExists(owner) onlyOwnerDoesNotExist(newOwner) { for  (uint256  i  =  0;  i  <  owners.length;  i++)  { if  (owners[i]  ==  owner)  { owners[i]  =  newOwner; break; } } isOwner[owner]  =  false; isOwner[newOwner]  =  true; OwnerRemoval(owner); OwnerAddition(newOwner); } }

# code_tokens docstring_tokens
def format_test():

    lang = 'cosqa'
    split = '<CODESPLIT>'

    FILE_DATA_DIR = './Dataset/{}/{}'.format('CoSQA', 'cosqa-all.json')
    #FILE_DATA_DIR = './SQL.txt'.format(lang)
    ROOT_DIR = './Dataset/{}/'.format('CoSQA')

    with open(FILE_DATA_DIR, "r", encoding="utf-8") as f:
        all_data = json.load(f)


    filtered_data = []
    # 过略掉负样本
    for item in all_data:
        if item['label'] == 1:
            filtered_data.append(item)


    X_train, X_val_test = train_test_split(filtered_data, test_size=0.15, random_state=1024)
    X_test, X_val = train_test_split(X_val_test, test_size=0.4, random_state=1024)

    for datadic in [['train',X_train], ['valid',X_val], ['test',X_test]]:

        lines = []
        print(datadic[0])
        ftype = datadic[0]
        data = datadic[1]
        for item in data:
            query = item['docstring_tokens'].replace('\n', ' ')
            code = item['code_tokens'].replace('\n', ' ')
            line = '1' + split + 'URL' + split + 'func_name' + split + query + split + code + '\n'
            lines.append(line)

        with open('{}{}.txt'.format(ROOT_DIR, ftype), 'w') as file:
            file.writelines(lines)



if __name__ == '__main__':
    format_test()