import os
import json

from sklearn.model_selection import train_test_split

# 将人工标注的数据转为 json格式

langs = ['SQL', 'Solidity', 'CoSQA']
file_types = ['train', 'valid', 'test']
# 格式
# "_id" "text", "title", "metadata"
#1<CODESPLIT>URL<CODESPLIT>func_name<CODESPLIT>Replace the owner with a new owner .<CODESPLIT>contract  c29479{ /** *  @dev  Replace  the  owner  with  a  new  owner *  @dev  Transaction  has  to  be  sent  by  wallet *  @param  owner  The  address  of  owner  to  be  replaced *  @param  newOwner  The  address  of  new  owner */ function  replaceOwner(address  owner,  address  newOwner) public onlyWallet onlyOwnerExists(owner) onlyOwnerDoesNotExist(newOwner) { for  (uint256  i  =  0;  i  <  owners.length;  i++)  { if  (owners[i]  ==  owner)  { owners[i]  =  newOwner; break; } } isOwner[owner]  =  false; isOwner[newOwner]  =  true; OwnerRemoval(owner); OwnerAddition(newOwner); } }

# code_tokens docstring_tokens
def format_test():

    lang = 'ruby'
    split = '<CODESPLIT>'


    #FILE_DATA_DIR = './SQL.txt'.format(lang)
    ROOT_DIR = './Dataset/csn/{}/'.format(lang)



    for datadic in [['train','train.jsonl'], ['valid','valid.jsonl'], ['test','test.jsonl']]:
        FILE_DATA_DIR = './Dataset/csn/{}/{}'.format(lang, datadic[1])
        ftype = datadic[0]
        data = datadic[1]

        all_data = []
        with open(FILE_DATA_DIR, 'r', encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                all_data.append(data)
                print(data)

        lines = []
        for item in all_data:
            query = item['docstring'].replace('\n', ' ').replace('\r', ' ')
            code = item['code'].replace('\n', ' ').replace('\r', ' ')
            line = '1' + split + 'URL' + split + 'func_name' + split + query + split + code + '\n'
            lines.append(line)

        with open('{}{}.txt'.format(ROOT_DIR, ftype), 'w') as file:
            file.writelines(lines)



if __name__ == '__main__':
    format_test()