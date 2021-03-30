# Recognize Chinese company name from text snippet 

## Install Requirements
`pip install -r requirements.txt`

## Prepare Models
We use [chinese_roberta_wwm_large_ext_pytorch](https://github.com/ymcui/Chinese-BERT-wwm)

## Train
The main training procedure is in `trainer.py`

Examples to start training are in `scripts/reproduce`.

Note that you may need to change `DATA_DIR`, `BERT_DIR`, `OUTPUT_DIR` to your own
dataset path, bert model path and log path, respectively.

## Evaluate and predict
`trainer.py` will automatically evaluate on dev set every `val_check_interval` epochs,
and save the topk checkpoints to `default_root_dir`.

To evaluate them, use `evaluate.py`

To predict, use `predict.py`


Key ideas to get this working are due to [this github](https://github.com/ShannonAI/mrc-for-flat-nested-ner).