import json
import string
from rouge import Rouge
import torch
import pandas as pd
from tqdm import tqdm
from transformers.models.t5 import T5ForConditionalGeneration, T5Tokenizer,T5Config
from transformers import RobertaTokenizer

from openprompt.plms import T5TokenizerWrapper


WrapperClass = T5TokenizerWrapper

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_config = T5Config.from_pretrained("./models/pre-trained/codet5-base-multi-sum")
model = T5ForConditionalGeneration.from_pretrained("./models/prompt_t5/", config=model_config, local_files_only=True)


tokenizer = RobertaTokenizer.from_pretrained("./models/pre-trained/codet5-base")
#tokenizer = T5Tokenizer.from_pretrained("./models/pre-trained/codet5-base-multi-sum")
rouge = Rouge()
model.to(DEVICE)
rouge1_score = 0
rouge2_score = 0
rougeL_score = 0
rouge_count = 0


def get_title(prefix, input_text, model):
    print(prefix + ": " + input_text)
    input_ids = tokenizer(input_text ,return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    summary_text_ids = model.generate(
        input_ids=input_ids["input_ids"].to(DEVICE),
        attention_mask=input_ids["attention_mask"].to(DEVICE),
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        length_penalty=1.2,
        top_k=20,
        top_p=0.9,
        max_length=20,
        min_length=5,
        num_beams=10,
    )
    title = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
    return title

generation_arguments = {
    "max_length": 20,
    "max_new_tokens": None,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.0,
    "num_beams": 10,
    "num_return_sequences":5
}


if __name__ == '__main__':

    body = "SELECT count(*) FROM routes WHERE dst_apid IN (SELECT apid FROM airports WHERE country = 'Canada') AND src_apid IN (SELECT apid FROM airports WHERE country = 'United States')"
    body1 = """def svg_to_image(string, size=None):
    if isinstance(string, unicode):
        string = string.encode('utf-8')
        renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(string))
    if not renderer.isValid():
        raise ValueError('Invalid SVG data.')
    if size is None:
        size = renderer.defaultSize()
        image = QtGui.QImage(size, QtGui.QImage.Format_ARGB32)
        painter = QtGui.QPainter(image)
        renderer.render(painter)
    return image"""
    title = get_title('summarize', body, model)
    print('title: {}'.format(title))
