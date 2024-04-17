When training models such as UniXcoder and CoCoSoDa, the sampling data need to be converted into fixed-format training data using convertGen2Jsonl\convertGen2Jsonl.py.

1) The first step is to perform data preprocessing.

2) In the second step, execute ``python 2_generation\qgen.py --lang SQL --prefix qgen --ques_per_passage 3`` to generate pseudo labels.

3) The third step is to execute ``python 3_negative\mine.py--lang=SQL --prefix=qgen --train_type gen --retrievers bm25 msmarco-distilbert-base-v3 msmarco-MiniLM-L-6-v3 -- retriever_score_functions none cos_sim cos_sim`` hard negative sampling.

4) Finally train the model ``python 4_retriever\training_1stage.py --lang SQL --train_type rel --prefix qgen --retrievers bm25 msmarco-distilbert-base-v3 msmarco-MiniLM-L-6-v3``.
