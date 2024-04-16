# # Conditional Generation with Prefix Tuning.
# In this tutorial, we do conditional generation with prefix tuning template.

# we use WebNLG as an example, as well. Note that the evaluation of generation result should be done
# by using the scripts provided by https://github.com/Yale-LILY/dart/tree/master/evaluation,
# Which we do not include in it.

import argparse
import os.path

import torch
from openprompt.data_utils import InputExample
from openprompt.prompts import PrefixTuningTemplate,SoftTemplate
from beir.datasets.data_loader import GenericDataLoader
from openprompt import PromptDataLoader
from transformers import AdamW
import random
from tqdm import tqdm
from openprompt.plms import T5TokenizerWrapper
from transformers import (AdamW, get_linear_schedule_with_warmup,
						  RobertaConfig, RobertaModel, RobertaTokenizer, T5Config, T5ForConditionalGeneration)
import numpy as np


parser = argparse.ArgumentParser("")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='t5')  # tested model are gpt2/t5
parser.add_argument("--model_name_or_path", default='t5-base')
args = parser.parse_args()
print(args)

from openprompt.data_utils.conditional_generation_dataset import WebNLGProcessor
dataset = {}
train_examples = []
valid_examples = []
DATA_PATH = './Dataset/'
gen_model_path = './models/prompt_t5'
lang = 'SQL'
train_path = os.path.join(DATA_PATH, lang, 'train')
valid_path = os.path.join(DATA_PATH, lang, 'valid')

train_corpus, train_queries, train_qrels = GenericDataLoader(
    train_path
).load(split="train")

valid_corpus, valid_queries, valid_qrels = GenericDataLoader(
    valid_path
).load(split="valid")

for qid, cidic in train_qrels.items():
    query = train_queries.get(qid)
    cid = list(cidic.keys())[0]
    code = train_corpus.get(cid)
    train_examples.append(InputExample(text_a=code.get('text'), tgt_text=query, guid=cid))

#sample_num = 3000
#random.seed(1024)
#train_examples = random.sample(train_examples, min(sample_num, len(train_examples)))

for qid , cidic in valid_qrels.items():
    query = train_queries.get(qid)
    cid = list(cidic.keys())[0]
    code = valid_corpus.get(cid)
    valid_examples.append(InputExample(text_a=code.get('text'), tgt_text=query, guid=cid))

sample_num = 500
valid_examples = random.sample(valid_examples, min(sample_num, len(valid_examples)))

dataset['train'] = train_examples
dataset['validation'] = valid_examples
dataset['test'] = valid_examples

# load a pretrained model, its tokenizer, its config, and its TokenzerWrapper by one function
#from openprompt.plms import load_plm:
#plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)
model_config = T5Config.from_pretrained("./models/pre-trained/codet5-base")
plm = T5ForConditionalGeneration.from_pretrained("./models/pre-trained/codet5-base", config=model_config)
tokenizer = RobertaTokenizer.from_pretrained("./models/pre-trained/codet5-base")
WrapperClass = T5TokenizerWrapper

# Instantiating the PrefixTuning Template !
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
# we can use a plain text as the default setting
# i.e.
# mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer)
# is equal to
# mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"mask"}')
mytemplate = PrefixTuningTemplate(model=plm,  tokenizer=tokenizer, text='Code: {"placeholder":"text_a"} {"special": "<eos>"} Summarization: {"mask"} ', using_decoder_past_key_values=False)
#mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, text='summarize: {"placeholder":"text_a"} {"mask"} ', initialize_from_vocab=True, num_tokens=50)
# define template
#mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, text='Code: {"placeholder":"text_a"} Summarization: {"mask"} ', initialize_from_vocab=True, num_tokens=50)

# To better understand how does the template wrap the example, we visualize one instance.
# You may observe that the example doesn't end with <|endoftext|> token. Don't worry, adding specific end-of-text token
# is a language-model-specific token. we will add it for you in the TokenizerWrapper once you pass `predict_eos_token=True`
wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
print(wrapped_example)


# Your can loop over the dataset by yourself by subsequently call mytemplate.wrap_one_example  and WrapperClass().tokenizer()
# but we have provide a PromptDataLoader for you.

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=32,
    batch_size=5,shuffle=True, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
    truncate_method="head")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=32,
    batch_size=5,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

test_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=32,
    batch_size=5,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

# load the pipeline model PromptForGeneration.
from openprompt import PromptForGeneration
use_cuda = True
prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=False, tokenizer=tokenizer, plm_eval_mode=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()



# Follow PrefixTuning（https://github.com/XiangLi1999/PrefixTuning), we also fix the language model
# only include the template's parameters in training.
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
{
    "params": [p for n, p in mytemplate.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in mytemplate.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    "weight_decay": 0.0,
},
]


optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8) #1e-8

from transformers.optimization import get_linear_schedule_with_warmup

tot_step  = len(train_dataloader)*50
scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

# We provide generation a generation metric, you can also define your own. Note that it's not directly comparable to WebNLG's scripts evaluation.
from openprompt.utils.metrics import generation_metric
# Define evaluate function
def evaluate(prompt_model, dataloader):
    generated_sentence = []
    groundtruth_sentence = []
    prompt_model.eval()

    prompt_model.load_state_dict(torch.load(os.path.join(gen_model_path, "pytorch_model.bin")))
    for step, inputs in enumerate(tqdm(dataloader)):
        if use_cuda:
            inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(inputs['tgt_text'])
    score = generation_metric(generated_sentence, groundtruth_sentence, "sentence_bleu")
    print("test_score", score, flush=True)
    return generated_sentence



generation_arguments = {
    "max_length": 32,
    "max_new_tokens": None,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 5,
    #"bad_words_ids": [[628], [198]]
}

# training and generation.
global_step = 0
tot_loss = 0
log_loss = 0
best_bleu = 0
best_loss = 1e6
for epoch in range(50):
    prompt_model.train()
    for step, inputs in enumerate(tqdm(train_dataloader)):
        global_step +=1
        this_epoch_best = False
        if use_cuda:
            inputs = inputs.cuda()
        loss = prompt_model(inputs)
        loss.backward()
        tot_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # save last checkpoint
    last_output_dir = os.path.join(gen_model_path, 'checkpoint-last')
    if not os.path.exists(last_output_dir):
        os.makedirs(last_output_dir)

    # Only save the model it-self
    model_to_save = prompt_model.module if hasattr(prompt_model, 'module') else prompt_model
    # eval
    generated_sentence = []
    groundtruth_sentence = []
    eval_loss = 0
    prompt_model.eval()
    for step, inputs in enumerate(tqdm(validation_dataloader)):
        if use_cuda:
            inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(inputs['tgt_text'])
    score = generation_metric(generated_sentence, groundtruth_sentence, "sentence_bleu")
    if score > best_bleu:
        output_model_file = os.path.join(gen_model_path, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        print('save best model bleu {}'.format(score))
        best_bleu = score
        this_epoch_best = True

    for step, inputs in enumerate(tqdm(validation_dataloader)):
        if use_cuda:
            inputs = inputs.cuda()
        with torch.no_grad():
            loss = prompt_model(inputs)
        eval_loss += loss.sum().item()


    if eval_loss < best_loss:
        output_model_file = os.path.join(gen_model_path, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        print('save best model ppl {}'.format(eval_loss))
        best_loss = eval_loss
        this_epoch_best = True
    # whether to stop
    if this_epoch_best:
        early_stop_count = 0
    else:
        early_stop_count += 1
        if early_stop_count == 3:
            print("early stopping!!!")
            break
    print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss) / ((epoch+1)*len(train_dataloader)), scheduler.get_last_lr()[0]), flush=True)

generated_sentence = evaluate(prompt_model, test_dataloader)


if not os.path.exists('./output/'):
    os.makedirs('./output/')

with open(f"./output/gen.txt",'w') as f:
    for i in generated_sentence:
        f.write(i+"\n")