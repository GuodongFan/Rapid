import random

from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset, PreTrainedTokenizer
from torch.utils.data import Dataset
import argparse
import torch
import json
import os
import math
CUR_PATH = os.getcwd()
print(CUR_PATH)
os.environ["WANDB_DISABLED"] = "true"


class LineByLineDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, examples: list, block_size: int):

        lines = examples

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)# add_special_tokens=True, truncation=True,
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

def main(lang, model_name):

    train_file1 = './Dataset/{}/train/corpus.jsonl'.format(lang)
    train_file2 = './Dataset/{}/train/queries.jsonl'.format(lang)
    eval_file = './Dataset/{}/valid/corpus.jsonl'.format(lang)
    eval_file2 = './Dataset/{}/valid/corpus.jsonl'.format(lang)
    model_dir = "./models/"

    print("lang {}".format(lang))

    train_sentences = []
    eval_setences = []
    with open(train_file1, 'r') as file, open(train_file2, 'r') as file2:

        lines = file.readlines()
        lines2 = file2.readlines()
        for idx, line in enumerate(lines):
            one_data = json.loads(line)
            #one_data_nl = json.loads(lines2[idx])

            #sentence_nl = one_data_nl['text']
            sentence = one_data['text']
            #train_sentences.append('<CLS> ' + sentence_nl + ' <SEP> ' + sentence)
            train_sentences.append(sentence)
            #train_sentences.append(sentence_nl)


    random.shuffle(train_sentences)

    print(train_sentences[:10])

    with open(eval_file, 'r') as file, open(eval_file2, 'r') as file2:

        lines = file.readlines()
        lines2 = file2.readlines()
        for idx, line in enumerate(lines):
            one_data = json.loads(line)
            one_data_nl = json.loads(lines2[idx])

            sentence_nl = one_data_nl['text']
            sentence = one_data['text']
            #eval_setences.append('<CLS> ' + sentence_nl + ' <SEP> ' + sentence)
            eval_setences.append(sentence)
            #eval_setences.append(sentence_nl)


    out_model_path = os.path.join(model_dir, 'mlm_model_{}'.format(lang))
    train_epoches = 5
    batch_size = 32

    # 这里不是从零训练，而是在原有预训练的基础上增加数据进行预训练，因此不会从 config 导入模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    dataset = LineByLineDataset(
        tokenizer=tokenizer,
        examples=train_sentences,
        block_size=128,

    )

    eval_dataset = LineByLineDataset(
        tokenizer=tokenizer,
        examples=eval_setences,
        block_size=128,
    )


    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)


    training_args = TrainingArguments(
        output_dir=out_model_path,
        overwrite_output_dir=True,
        num_train_epochs=train_epoches,
        per_device_train_batch_size=batch_size,
        save_steps=2000,
        save_total_limit=3,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    tokenizer.save_pretrained(out_model_path)
    trainer.save_model(out_model_path)
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    print(out_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='SQL')
    parser.add_argument("--model", default='microsoft/codebert-base')
    args = parser.parse_args()

    main(args.lang, args.model)