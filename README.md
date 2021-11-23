# chaii 4th place solution
Team member: Yuchen Wang (NYU Shanghai), Zhengye Zhu (Peking University).

This is the implementation of the 4th place solution of the [chaii - Hindi and Tamil Question Answering](https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering) competition at kaggle.

Our solution write-up: https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/discussion/287911.

Dataset we made (not involved in the final submission): [hi/ta parsed wiki](https://www.kaggle.com/zacchaeus/chaii-tfds-wiki), [SQuAD 2.0 in Tamil](https://www.kaggle.com/zacchaeus/chaii-tfds-wiki), [cleaned chaii dataset](https://www.kaggle.com/zacchaeus/chaiitrain0917).

## To reproduce our result:
1. The environment is the same as a [Kaggle Docker](https://github.com/Kaggle/docker-python). Install dependencies with `pip install -r requirements.txt`. You will need a single RTX3090 or A10.
2. To leverage zero-shot transferability, finetune RemBERT, InfoXLM, Muril, XLM-R on SQuAD 2.0 with `finetune.py`.
An example of finetuning Muril:
            
            python finetune.py -u \
            --model_checkpoint google/muril-large-cased \
            --train_path <path to data>/train-v2.0.json \
            --max_length 512 \
            --doc_stride 128 \
            --epochs 2 \
            --batch_size 4 \
            --accumulation_steps 8 \
            --lr 1e-5 \
            --weight_decay 0.01 \
            --warmup_ratio 0.2 \
            --seed 42 \
            --dropout 0.1
            
      Substitute `model_checkpoint` with corresponding Huggingface pre-trained checkpoint for other models. Set epochs = 3 for RemBERT, InfoXLM, XLM-R, leaving other hyper-parameters the same.

3. As decribed in [our solution write-up](https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/discussion/287911), we trained models with corss-validation or with all data. You can train 5-fold models on the chaii + XQuAD + MLQA dataset with `train-cv.py` OR train with all data with `train-all.py`. Please first download our cleaned data [here](https://www.kaggle.com/zacchaeus/chaiitrain0917).
    * An example of training 5 folds Muril, substitute `model_checkpoint` for the others:
            ```
            python -u train-native-stepeval.py \
            --model_checkpoint google/muril-large-cased \
            --train_path <path to data>/merged0917.csv \
            --max_length 512 \
            --doc_stride 128 \
            --epochs 3 \
            --batch_size 4 \
            --accumulation_steps 1 \
            --lr 1e-5 \
            --optimizer adamw \
            --weight_decay 0.0 \
            --scheduler cosann \
            --warmup_ratio 0.1 \
            --dropout 0.1 \
            --eval_steps 1000 \
            --metric nonzero_jaccard_per \
            --downext \
            --seed 42
            ```
    * An example of training Muril with all data, substitute `model_checkpoint` for the others:
            ```
            python -u train-useall.py \
            --model_checkpoint google/muril-large-cased \
            --train_path <path to data>/merged0917.csv \
            --max_length 512 \
            --doc_stride 128 \
            --epochs 3 \
            --batch_size 4 \
            --accumulation_steps 1 \
            --lr 1e-5 \
            --weight_decay 0.0 \
            --warmup_ratio 0.1 \
            --seed 42 \
            --dropout 0.1 \
            --downsample 0.5
            ```
    * Although we didn't find the translated SQuAD dataset useful, you may try to train on it with `train-enta.py` on [SQuAD 2.0 in Tamil](https://www.kaggle.com/zacchaeus/chaii-tfds-wiki).
4. Infer with ensembling and post-processing: https://www.kaggle.com/zacchaeus/chaii-infer-blend-postpro-4models.
