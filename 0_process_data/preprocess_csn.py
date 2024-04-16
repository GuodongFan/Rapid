import json
import os
from sklearn.model_selection import train_test_split


DIR = '../Dataset/'
#lang = 'javascript'
file_types = ['train', 'valid', 'test']
file_type = 'train'

lang_filter = {'advtest': 100}
#lang_filter = {'python': 100}

def write_file(lang, file_type, task_type, data):
    out_dir_base = os.path.join(DIR, lang, file_type + '_' + task_type + '.jsonl')
    with open(out_dir_base, 'w') as file:
        file.writelines(data)

def write_file_txt(lang, file_type, task_type, data):
    out_dir_base = os.path.join(DIR, lang, file_type + '_' + task_type + '.txt')
    with open(out_dir_base, 'w') as file:
        file.writelines(data)


def merge(lang, file_types):
    all_data = []
    all_project = {}
    for file_type in file_types:
        file_dir = os.path.join(DIR, lang, file_type + '.jsonl')
        with open(file_dir, encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                try:
                    js_list = json.loads(line)
                except:
                    print('error')
                    continue

                try:
                    pj_name = js_list['repo']
                except:
                    # 使用split方法分割URL
                    parts = js_list['url'].split('/')

                    # 从分割后的列表中提取所有者和仓库名称
                    owner = parts[3]
                    repository = parts[4]
                    pj_name = owner + '/' + repository
                    js_list['repo'] = pj_name
                all_data.append(json.dumps(js_list)+'\r\n')

                if all_project.get(pj_name) == None:
                    all_project[pj_name] = 1
                else:
                    all_project[pj_name] = all_project[pj_name] + 1



    return all_data, all_project

def process(lang):
    train_data, train_project = merge(lang, ['train'])
    test_data, test_project = merge(lang, ['test'])


    proj_max = 0
    proj_name = ''
    for proj, count in test_project.items():
        if proj in train_project:
            print(proj)
            continue
        if count > proj_max:
            proj_name = proj
            proj_max = count
            print(proj_name)
            print(count)

    print(proj_name)
    print(proj_max)


    short_name = 'airflow'
    proj_name = f'apache/{short_name}'

    data = []
    lines_data = []
    for line in test_data:
        #print(line)

        item = json.loads(line)
        if item['repo'] == proj_name:
            data.append(line)

            split = '<CODESPLIT>'
            query = item['docstring'].replace('\n', ' ').replace('\r', ' ')
            code = item['function'].replace('\n', ' ').replace('\r', ' ')
            temp = '1' + split + 'URL' + split + 'func_name' + split + query + split + code + '\n'
            lines_data.append(temp)

    write_file(lang+'/'+short_name, 'repo', 'test', data)

    write_file_txt(lang+'/'+short_name, 'repo', 'test', lines_data)

    print('')

if __name__ == '__main__':
    print(lang_filter)
    for key, val in lang_filter.items():
        print('key {} val {}'.format(key, val))
        process(key)