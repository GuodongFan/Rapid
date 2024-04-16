from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from torch.utils.data import DataLoader
import json
import argparse
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, InformationRetrievalEvaluator


'''from modeling_distilbert import DistilBertLMHeadModel
# Please run the following command to get the extension:
# git clone https://gist.github.com/1f0e1f0ce119456284c0af048ba097a7.git
# mv 1f0e1f0ce119456284c0af048ba097a7/modeling_distilbert.py ./
from transformers import AutoModelForCausalLM
from transformers import DistilBertConfig
AutoModelForCausalLM.register(DistilBertConfig, DistilBertLMHeadModel)'''

def main(lang, model_name):

    train_file = './Dataset/{}/corpus.jsonl'.format(lang)
    eval_file = './Dataset/{}/valid/corpus.jsonl'.format(lang)
    model_dir = "./models/"
    # Define your sentence transformer model using CLS pooling
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


    train_sentences = []
    with open('./Dataset/{}/corpus.jsonl'.format(lang), 'r') as file:

        lines = file.readlines()
        for line in lines:
            one_data = json.loads(line)
            sentence = one_data['text']
            train_sentences.append(sentence[:256])
        #datas = json.load(file)
        #for data in datas:
        #    sentence = data['review_raw']
        #    train_sentences.append(sentence)

    test_sentences = []

    test_path = "{}/test/".format('./Dataset/CoSQA')

    # Create the special denoising dataset that adds noise on-the-fly
    train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)

    # DataLoader to batch your data
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Use the denoising auto-encoder loss
    train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=True)

    # Call the fit method
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=5,
        weight_decay=0,
        scheduler='constantlr',
        optimizer_params={'lr': 3e-5},
        show_progress_bar=True
    )

    model__format = './models/tsdae_{}_CoCoSoDa_model'.format(lang)
    model.save(model__format)

    corpus, test_queries, test_qrels = GenericDataLoader(
        test_path, prefix=''
    ).load(split="test")

    test_evaluator = InformationRetrievalEvaluator(test_queries, corpus, test_qrels)
    print(model.evaluate(test_evaluator))
    print(model__format)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='CoSQA')
    parser.add_argument("--model", default='/data1/fgd/workplace/models/bert-base-uncased/')
    args = parser.parse_args()

    main(args.lang, args.model)