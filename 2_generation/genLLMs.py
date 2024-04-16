import os.path
from util import *

from beir.datasets.data_loader import GenericDataLoader
import openai
from tqdm import tqdm
import time

# Load your API key from an environment variable or secret management service
openai.api_key = "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"

lang = 'SQL'
ROOT_PATH = './Dataset/'
data_path = '{}{}/'.format(ROOT_PATH, lang)
corpus = GenericDataLoader(data_path).load_corpus()
# response = openai.Completion.create(model="text-davinci-003", prompt="Say this is a test", temperature=0, max_tokens=7)

prefix='gpt3'
qrels = {}
queries = {}

qrels_name = prefix + '_qrels'
count = 0
for corpus_id, one_code in tqdm(corpus.items()):
    if int(corpus_id[2:]) < 740 and int(corpus_id[2:]) < 755:
        continue

    prompt = "summarize: " + one_code['text']
    model_engine = "text-davinci-003"
    #20230312
    try:
        completions = openai.Completion.create(
            engine = model_engine,
            prompt = prompt,
            max_tokens = 50,
            n = 5,
            stop = None,
            temperature=0.5,
        )
    except Exception as e:
        print(str(e))
        print('*****write {}*****'.format(corpus_id))
        write_to_json(output_file=os.path.join(ROOT_PATH, lang, prefix + '_queries_temp.jsonl'), data=queries)
        write_to_tsv(output_file=os.path.join(ROOT_PATH, lang, qrels_name, 'train_temp.tsv'), data=qrels)
        exit(0)

    for choice in completions.choices:
        count += 1
        message = choice.text
        query_id = prefix + "_" + str(count)
        queries[query_id] = message
        qrels[query_id] = {corpus_id: 1}

        print(message)

    if count % 100 == 0:
        print('*****write {}*****'.format(corpus_id))
        write_to_json(output_file=os.path.join(ROOT_PATH, lang, prefix + '_queries_temp.jsonl'), data=queries)
        write_to_tsv(output_file=os.path.join(ROOT_PATH, lang, qrels_name, 'train_temp.tsv'), data=qrels)
    #time.sleep(2)

if not os.path.exists(os.path.join(ROOT_PATH, lang, qrels_name)):
    os.makedirs(os.path.join(ROOT_PATH, lang, qrels_name))

write_to_json(output_file=os.path.join(ROOT_PATH, lang, prefix + '_queries.jsonl'), data=queries)
write_to_tsv(output_file=os.path.join(ROOT_PATH, lang, qrels_name, 'train.tsv'), data=qrels)