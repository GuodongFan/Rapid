import numpy as np
from beir.datasets.data_loader import GenericDataLoader

from torchmetrics.text.rouge import ROUGEScore
rougeScore = ROUGEScore()
from bert_score import score

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk


def get_bleu(recover, reference):
    return sentence_bleu([reference.split()], recover.split(), smoothing_function=SmoothingFunction().method4,)

def selectBest(sentences):
    selfBleu = [[] for i in range(len(sentences))]
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            score = get_bleu(s1, s2)
            selfBleu[i].append(score)
    for i, s1 in enumerate(sentences):
        selfBleu[i][i] = 0
    idx = np.argmax(np.sum(selfBleu, -1))
    idx2 = np.argsort(-np.sum(selfBleu, -1))
    return sentences[idx], np.array(sentences)[idx2]


v = ['SELECT COUNT(support_rate) FROM candidate ORDER BY support_rate DESC',
     'SELECT support_rate FROM candidate ORDER BY support_rate DESC.',
     'selects the minimum support rate of 3 candidate items',
     'selects the minimum support rate of 3 candidate ']

print(selectBest(v))