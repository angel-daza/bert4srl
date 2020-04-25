# bert4srl
This code shows a **simple** way to Fine-Tune BERT on the task of Semantic Role Labeling. 
It is mostly meant to show a simple way of finetuning BERT using SRL as an example.  

## Requirements

* Python version >= 3.6.6
* Hugging-Face Transformers >= 2.2.0


## Data

This model works with any dataset in CoNLL09 Format (see example at data/trial.conll). 

We include the Trial Data for you to be able to test the code... 
however, note that no matter how long you train with this data, 
the output will be rubbish. You can create your own dataset following
the CoNLL format or, to reproduce the results on the SRL Shared Task, 
you should buy access to the CoNLL-09 Part 2 dataset which is part 
of [LDC Catalog](https://catalog.ldc.upenn.edu/LDC2012T04)


### Usage

#### Pre-processing

```
python pre_processing/conll2json.py \
	--source_file data/CoNLL2009-ST-English-trial.txt \
	--output_file data/Trial_EN.jsonl \
	--src_lang "<EN>" \
	--token_type CoNLL09_Token
```

#### Train a Model

```
python train.py --train_path data/Trial_EN.jsonl \
	--dev_path data/Trial_EN.jsonl \
	--save_model_dir saved_models/TRIAL_BERT
```

#### Make Predictions

```
python predict.py -m saved_models/TRIAL_BERT --epoch 1 --test_path data/Trial_EN.jsonl
```