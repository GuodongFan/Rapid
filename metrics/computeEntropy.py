from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline
import torch
import numpy as np
from beir.datasets.data_loader import GenericDataLoader

model = RobertaForMaskedLM.from_pretrained('microsoft/codebert-base-mlm')
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base-mlm')

code_example = "if (x is not None) <mask> (x>1)"
query = "Show the name of teachers aged either 32 or 33"

corpus, gen_queries, gen_qrels = GenericDataLoader(
    './Dataset/SQL/valid/', prefix=''
).load(split="valid")

#query = "Give the names of poker players who have earnings above 300000."
code = "SELECT Name FROM teacher WHERE Age = 32 OR Age = 33"



def get_prediction(query_mask, code_tokens, query_ids, code_ids, query_tokens):

    masked_position = (query_ids.squeeze() == tokenizer.mask_token_id).nonzero()
    masked_pos = [mask.item() for mask in masked_position]

    with torch.no_grad():
        input_ids = torch.cat([query_ids, code_ids], dim=1)
        output = model(input_ids)

    last_hidden_state = output[0].squeeze()

    ret_list = []
    for index, mask_index in enumerate(masked_pos):
        token = query_tokens[mask_index-1]
        token_id = tokenizer.convert_tokens_to_ids(token)
        probs = torch.nn.functional.softmax(last_hidden_state[mask_index])
        token_prob = probs[token_id]
        return token_prob
        ret_list.append(token_prob)
    return ret_list


def computeNaturalless(query, code):
    query = query + ' '
    code = ' ' + code
    fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)

    # query_fields = query.split()
    query_fields = tokenizer.tokenize(query)
    code_fields = tokenizer.tokenize(code)
    code_fields.insert(0, tokenizer.sep_token)

    query_fields_ids = tokenizer.encode(query_fields, return_tensors='pt')
    code_fields_ids = tokenizer.encode(code_fields, return_tensors='pt')
    query_mask = ["<mask>" for _ in range(len(query_fields))]
    query_mask = tokenizer.tokenize(' '.join(query_mask).strip())
    query_mask_ids = tokenizer.encode(query_mask, return_tensors='pt')
    mask_len = len(query_fields)

    range_ids = [i for i in range(mask_len)]

    choices = [id for id in range_ids if id%2==0]
    #choices = np.random.choice(range_ids, size=int((mask_len)/2), replace=False)
    choices = []

    for idx in choices:
        query_mask_ids[0, idx] = query_fields_ids[0, idx]

    probs = []
    query_str = tokenizer.convert_tokens_to_string(query_mask)
    input_example = query_str + " <pad> " + code
    #probs = get_prediction(query_mask, code_fields, query_mask_ids, code_fields_ids, query_fields)
    #log_probs = np.log2(probs)
    #shang = -1 * np.sum(probs * log_probs, axis=0)/len(log_probs)
    #print(shang)
    #return shang
    for idx, q_field in enumerate(range(len(query_fields))):
        query_str = tokenizer.convert_tokens_to_string(query_mask)

        input_example = query_str + " <pad> " + code

        prob = get_prediction(query_mask, code_fields, query_mask_ids, code_fields_ids, query_fields)

        query_mask_ids[0, idx] = query_fields_ids[0, idx]
        #print(prob)
        probs.append(prob)

    log_probs = np.log2(probs)
    ce = -1/ len(log_probs) * np.sum(log_probs, axis=0)
    #shang = -1 * np.sum(probs * log_probs, axis=0) / len(log_probs)
    print(ce)
    return ce

index = 0
base_shang = 0
count = 0

'''for qid, cidic in gen_qrels.items():
    for cid, _ in cidic.items():
        query = gen_queries.get(qid)
        code = corpus.get(cid)
        print('qid ', qid, ' cid ', cid)
        print(query)
        print(code)
        computeNaturalless(query, code.get('text'))
        print("***********")'''

query = gen_queries.get('q_0')
query = query
for cid, code in corpus.items():
    shang = computeNaturalless(query, code.get('text'))
    if index == 0:
        base_shang = shang
    elif base_shang >= shang:
        count+=1
    index += 1

print(count)