import json
import string
from rouge import Rouge
import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers.models.t5 import T5ForConditionalGeneration, T5Tokenizer,T5Config
from transformers import RobertaTokenizer

from openprompt.plms import T5TokenizerWrapper
from openprompt import PromptDataLoader, PromptForGeneration
from openprompt.plms import T5TokenizerWrapper
from openprompt.prompts import PrefixTuningTemplate,SoftTemplate
from openprompt.data_utils import InputExample

WrapperClass = T5TokenizerWrapper

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_config = T5Config.from_pretrained("./models/pre-trained/codet5-base")
#model = T5ForConditionalGeneration.from_pretrained("t5-base", config=model_config)
model = T5ForConditionalGeneration.from_pretrained("./models/prompt_t5/", config=model_config, local_files_only=True)
tokenizer = RobertaTokenizer.from_pretrained("./models/pre-trained/codet5-base")
model.to(DEVICE)

rouge = Rouge()
rouge1_score = 0
rouge2_score = 0
rougeL_score = 0
rouge_count = 0




generation_arguments = {
    "max_length": 20,
    "max_new_tokens": None,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.0,
    "num_beams": 20,
    "num_return_sequences":5
}


if __name__ == '__main__':

    #model = T5ForConditionalGeneration.from_pretrained("t5-base", config=model_config)
    plm = T5ForConditionalGeneration.from_pretrained("./models/prompt_t5/", config=model_config, local_files_only=True)

    promptTemplate = SoftTemplate(model=plm, tokenizer=tokenizer,
                                  text='summarize: {"placeholder":"text_a"} {"mask"}', initialize_from_vocab=True,
                                  num_tokens=20)
    promptTemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer,
                                      text='Code: {"placeholder":"text_a"} {"special": "<eos>"} Summarization: {"mask"} ',
                                      using_decoder_past_key_values=False)
    # get model
    model = PromptForGeneration(plm=plm, template=promptTemplate, freeze_plm=False, tokenizer=tokenizer,
                                plm_eval_mode=True)
    model.load_state_dict(torch.load(os.path.join('./models/prompt_t5/', "pytorch_model.bin")))
    model.to(DEVICE)
    body = """SELECT mID FROM Rating EXCEPT SELECT T1.mID FROM Rating AS T1 JOIN Reviewer AS T2 ON T1.rID = T2.rID WHERE T2.name = 'Brittany Harris'"""
    dataset = {}
    dataset['test'] = []
    input_example = InputExample(text_a=body, text_b='')
    dataset['test'].append(input_example)
    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=promptTemplate, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=32,
                                       batch_size=5, shuffle=False, teacher_forcing=False, predict_eos_token=True,
                                       truncate_method="head")



    for step, inputs in enumerate(test_dataloader):
        inputs = inputs.cuda()
        _, output_sentence = model.generate(inputs, **generation_arguments)
        print(output_sentence)