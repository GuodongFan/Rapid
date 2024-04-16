from nlgeval import compute_metrics
from nlgeval import compute_individual_metrics
from rouge import Rouge
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nlgeval import NLGEval
import argparse

nlgeval = NLGEval(no_skipthoughts=True,
                               no_glove=True)

parser = argparse.ArgumentParser()

# outdated parameters
parser.add_argument("--ground_truth", default='./data/output/gd.out', type=str, required=False)
parser.add_argument("--predict", default='./data/output/gen.out', type=str, required=False)
parser.add_argument("--trans", default=True, type=bool, required=False)
# print arguments
args = parser.parse_args()


# 转换格式
if args.trans:
    with open('{}'.format(args.predict)) as hypo, open('{}'.format(args.ground_truth)) as ref, open('{}_'.format(args.predict), "w") as f, open('{}_'.format(args.ground_truth), "w") as f1:
        print('preprocess')
        hypo_lines = hypo.readlines()
        ref_lines = ref.readlines()
        for idx, hypo_line in enumerate(hypo_lines):
            cols = hypo_line.strip().split('\t')
            if len(cols) == 1:
                hypo = cols[0]
            else:
                hypo = cols[1]

            cols = ref_lines[idx].strip().split('\t')
            if len(cols) == 1:
                refer = cols[0]
            else:
                refer = cols[1]

            f.write(hypo + '\n')
            f1.write(refer + '\n')



smooth = SmoothingFunction()
metrics_dict = compute_metrics(hypothesis='{}_'.format(args.predict),
                               references=['{}_'.format(args.ground_truth)])


def compute_metrics1(predictions, labels):
    decoded_preds = predictions 
    # Replace -100 in the labels as we can't decode them.
    decoded_labels = labels
 
    
    # 字符级别
    #decoded_preds = [" ".join((pred.replace(" ", ""))) for pred in decoded_preds]
    #decoded_labels = [" ".join((label.replace(" ", ""))) for label in decoded_labels]
    # 词级别，分词
    # decoded_preds = [" ".join(jieba.cut(pred.replace(" ", ""))) for pred in decoded_preds]
    # decoded_labels = [" ".join(jieba.cut(label.replace(" ", ""))) for label in decoded_labels]
    rouge = Rouge()
    labels_lens = len(labels)
 
 
    total = 0
 
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
    for decoded_label, decoded_pred in zip(decoded_labels, decoded_preds):
        total += 1
        scores = rouge.get_scores(hyps=decoded_pred, refs=decoded_label)
        rouge_1 += scores[0]['rouge-1']['f']
        rouge_2 += scores[0]['rouge-2']['f']
        rouge_l += scores[0]['rouge-l']['f']
        bleu += sentence_bleu(
            weights=(1,0,0,0),
            references=[decoded_label.split(' ')],
            hypothesis=decoded_pred.split(' '),
            smoothing_function=SmoothingFunction().method1
        )
    bleu /= len(decoded_labels)
    rouge_1 /= total
    rouge_2 /= total
    rouge_l /= total
    result = {'rouge-1': rouge_1, 'rouge-2': rouge_2, 'rouge-l': rouge_l}
    print(result)
    # 测试平均与分别计算是否一致
    result2 = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    print(result2)
    print(bleu)
    # result = {'rouge-1': result['rouge-1']['f'], 'rouge-2': result['rouge-2']['f'], 'rouge-l': result['rouge-l']['f']}
 
    result = {key: value * 100 for key, value in result.items()}
    result["gen_len"] = np.mean(labels_lens)
    result["bleu"] = bleu * 100
    return result

def get_bleu(reference, hypothesis):
    BLEUscore = sentence_bleu([reference.split(' ')], hypothesis.split(' '), weights = (1, 0, 0, 0), smoothing_function=smooth.method1)
    return BLEUscore

def get_f1(bleu, rouge):
    f1=(2*bleu*rouge)/(bleu+rouge)
    return f1


rouge = Rouge()
rouge1_score = 0
rouge2_score = 0
rougeL_score = 0
bleu1_score = 0
f1_score = 0

hyps = []
refs = []
with open(args.predict) as hypo, open(args.ground_truth) as ref:
    hypo_lines = hypo.readlines()
    ref_lines = ref.readlines()
    for idx, hypo_line in enumerate(hypo_lines):
        hypo = hypo_line
        refer = ref_lines[idx]
        hyps.append(hypo)
        refs.append(refer)
        #metric = compute_individual_metrics(hypo_line, ref_lines[idx], no_skipthoughts=True, no_glove=True)
        scores = rouge.get_scores([hypo], [refer])
        rouge1_score += scores[0]['rouge-1']['f']
        rouge2_score += scores[0]['rouge-2']['f']
        rougeL_score += scores[0]['rouge-l']['f']
        bleu1_score += get_bleu(refer, hypo)
print(rouge1_score/(idx+1))
print(rouge2_score/(idx+1))
print(rougeL_score/(idx+1))
print(bleu1_score/(idx+1))
#print(get_f1((bleu1_score/(idx+1)), (rouge1_score/(idx+1))))
print(compute_metrics1(hyps, refs))
