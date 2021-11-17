# chaii-4th-place-solution
This is the source code of the 4th place solution of the [chaii - Hindi and Tamil Question Answering](https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering) competition at kaggle.
Our solution write-up: https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/discussion/287911.
Inference kernel: https://www.kaggle.com/zacchaeus/chaii-infer-blend-postpro-4models.
Dataset we made (not involved in the final submission): [hi/ta parsed wiki] (https://www.kaggle.com/zacchaeus/chaii-tfds-wiki), [SQuAD 2.0 in Tamil] (https://www.kaggle.com/zacchaeus/chaii-tfds-wiki), [cleaned chaii dataset] (https://www.kaggle.com/zacchaeus/chaiitrain0917)
## The training pipeline:
1. Finetune RemBERT, InfoXLM, Muril, XLM-R on SQuAD 2.0 with `finetune.py`.
2. Train 5 folds on the chaii dataset with `train-cv.py`. OR Train with all data with `train-all.py`.
