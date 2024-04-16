import numpy as np
import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
DEVICE = 'cuda:0'

class Perplexity_Checker(object):
    def __init__(self, MODEL_PATH=None, MODEL_NAME=None, device='cuda'):
        if MODEL_PATH:
            self.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
            self.model = BertForMaskedLM.from_pretrained(MODEL_PATH)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
            self.model = BertForMaskedLM.from_pretrained(MODEL_NAME)
        self.model.to(device)
        self.model.eval()
        self.DEVICE = device

    def add_device(self, DEVICE):
        self.DEVICE = DEVICE
        self.model.to(DEVICE)

    def sentence_preprocese(self, text):
        tokenized_text = np.array(self.tokenizer.tokenize(text))
        find_sep = np.argwhere(tokenized_text == '[SEP]')
        segments_ids = np.zeros(tokenized_text.shape, dtype=int)
        if find_sep.size == 1:
            start_point = 1
        else:
            start_point = find_sep[0, 0] + 1
            segments_ids[start_point:] = 1

        end_point = tokenized_text.size - 1

        tokenized_text = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        masked_texts = []
        for masked_index in range(start_point, end_point):
            new_tokenized_text = np.array(tokenized_text, dtype=int)
            new_tokenized_text[masked_index] = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
            masked_texts.append(new_tokenized_text)

        segments_ids = np.tile(segments_ids, (end_point - start_point, 1))

        return masked_texts, segments_ids, start_point, end_point, tokenized_text[start_point:end_point]

    def perplexity(self, text):
        indexed_tokens, segments_ids, start_point, end_point, real_indexs = self.sentence_preprocese(text)

        tokens_tensor = torch.LongTensor(indexed_tokens)
        segments_tensors = torch.LongTensor(segments_ids)

        tokens_tensor = tokens_tensor.to(self.DEVICE)
        segments_tensors = segments_tensors.to(self.DEVICE)

        with torch.no_grad():
            outputs = self.model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = torch.softmax(outputs[0], -1)

        total_perplexity = 0
        for i, step in enumerate(range(start_point, end_point)):
            total_perplexity += np.log(predictions[i, step, real_indexs[i]].item())

        total_perplexity = -total_perplexity / (end_point - start_point)
        return total_perplexity


if __name__ == '__main__':

    # 模型名字
    MODEL_NAME = "bert-base-uncased"

    text_formatter = lambda x: "[CLS] {} [SEP]".format(x)
    pchecker = Perplexity_Checker(MODEL_NAME=MODEL_NAME, device='cuda')

    # 计算困惑度，按理来说困惑度越低越好，但bert-chinese实在太烂了……
    text1 = 'SELECT FIRST NAME FROM actor WITH COUNT 1 - N ROM'
    text2 = 'selects a list of support rates in descending order'
    text3 = 'SELECT BOLDS FROM accounts T2 T3. name AS name'
    text4 = 'SELECT name origin'
    print(pchecker.perplexity(text_formatter(text1)))
    print(pchecker.perplexity(text_formatter(text2)))
    print(pchecker.perplexity(text_formatter(text3)))
    print(pchecker.perplexity(text_formatter(text4)))