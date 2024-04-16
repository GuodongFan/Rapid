import json
import time
import pandas as pd

from openprompt.data_utils import InputExample

def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename, args):
    """Read examples from filename."""
    if args.add_task_prefix:
        task_prefix = f"Generating comments for {args.lang}: "
    else:
        task_prefix = ""

    if args.add_lang_ids:
        language_prefix = "<en> "
    else:
        language_prefix = ""

    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            # print(js.keys())
            # try:
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            # except:

            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            # code = js['code'].replace('\n', ' ').strip().replace('\t', ' ')
            # nl = js['docstring'].replace('\n', ' ').replace('\t', ' ').strip()
            examples.append(
                Example(
                    idx=idx,
                    source=task_prefix + code,
                    target=language_prefix + nl,
                )
            )

    return examples


class InputFeatures(object):
    """A single training/test features for an example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    # collect texts
    codes = []
    target_nl = []
    for example_id, example in enumerate(examples):
        codes.append(example.source)

        if stage == "test":
            target_nl.append("None")
        else:
            target_nl.append(example.target)

    # begin tokenizing
    encoded_codes = tokenizer(
        codes, padding=True, verbose=False, add_special_tokens=True,
        truncation=True, max_length=args.max_source_length, return_tensors='pt')

    encoded_targets = tokenizer(
        target_nl, padding=True, verbose=False, add_special_tokens=True,
        truncation=True, max_length=args.max_source_length, return_tensors='pt')

    return {'source_ids': encoded_codes['input_ids'], 'target_ids': encoded_targets['input_ids'],
            'source_mask': encoded_codes['attention_mask'], 'target_mask': encoded_targets['attention_mask']}


def read_prompt_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            # code = js['code'].replace('\n', ' ').strip()
            # nl = js['docstring'].replace('\n', ' ').strip()
            examples.append(
                InputExample(
                    guid=idx,
                    text_a=code,
                    tgt_text=nl,
                )
            )

    return examples


def read_prompt_examples_pd(filename):
    """Read examples from filename."""
    examples = []
    df = pd.read_csv(filename, header=0)
    for i, data in df.iterrows():
        code = data[1]
        nl = data[2]
        examples.append(
            InputExample(
                guid=i,
                text_a=code,
                tgt_text=nl,
            )
        )
    return examples


def read_finetune_examples_pd(filename):
    """Read examples from filename."""
    if True:
        task_prefix = f"Summarization: "
    else:
        task_prefix = ""

    examples = []
    df = pd.read_csv(filename, header=0)
    for i, data in df.iterrows():
        code = data[1]
        nl = data[2]
        examples.append(
            Example(
                idx=i,
                source=code,  # task_prefix + code,
                target=nl,
            )
        )
    return examples