export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false
DATA_DIR="/data/ai_group/mrc-ner-company/datasets/zh_msra"
BERT_DIR="/data/ai_group/mrc-ner-company/chinese_roberta_wwm_large_ext_pytorch"
SPAN_WEIGHT=0.1
DROPOUT=0.2
LR=8e-6
MAXLEN=128

OUTPUT_DIR="/data/ai_group/mrc-ner-company/train_logs/zh_msra_company/zh_msra_bertlarge_lr${LR}20200913_dropout${DROPOUT}_bsz16_maxlen${MAXLEN}"

mkdir -p $OUTPUT_DIR

#change:  --gpus="0,1,2,3" \
#delete: --hard_span_only \

python ../../trainer.py \
--chinese \
--data_dir $DATA_DIR \
--bert_config_dir $BERT_DIR \
--max_length $MAXLEN \
--batch_size 4 \
--gpus=1 \
--precision=16 \
--progress_bar_refresh_rate 1 \
--lr ${LR} \
--distributed_backend=ddp \
--val_check_interval 0.5 \
--accumulate_grad_batches 1 \
--default_root_dir $OUTPUT_DIR \
--mrc_dropout $DROPOUT \
--max_epochs 20 \
--weight_span $SPAN_WEIGHT \
--span_loss_candidates "pred_and_gold"
