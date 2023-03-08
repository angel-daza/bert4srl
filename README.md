# bert4srl
This code shows a **simple** way to Fine-Tune BERT on the task of Semantic Role Labeling. 
It is mostly meant to show a simple way of finetuning BERT using SRL as an example.  

## Requirements

* Python version >= 3.8
* Hugging-Face Transformers >= 4.17.0


## Data

This model was tested with [Universal Proposition Banks](https://github.com/UniversalPropositions/UP-1.0) dataset. Further compatibility for CoNLL-05, CoNLL-09, CoNLL-12 (all of them are licensed datasets) can be easily added by creating the appropriate objects for data pre_processing.


### Usage

#### Pre-processing

```
python pre_processing/conll2json.py \
            --source_file data/en_ewt-up-dev.conllu \
            --output_file data/en_ewt-up-dev.jsonl \
            --src_lang "<EN>" \
            --token_type EN_CoNLLUP_Token
```

#### Train a Model

```
python3 finetune_bert.py --train_path data/en_ewt-up-train.jsonl --dev_path data/en_ewt-up-dev.jsonl --save_model_dir saved_models/MBERT_SRL \
        --epochs 10 --batch_size 16 --info_every 100 --bert_model bert-base-multilingual-cased
```

#### Make Predictions

```
python3 predict.py -m saved_models/EN_BERT_SRL --epoch 10 --test_path data/en_ewt-up-test.jsonl
```