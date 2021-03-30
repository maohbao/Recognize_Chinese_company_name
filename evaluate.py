# encoding: utf-8


import os
from pytorch_lightning import Trainer

from trainer import BertLabeling


def evaluate(ckpt, hparams_file):
    """main"""

    trainer = Trainer(gpus=[0], distributed_backend="ddp")  ######  gpus=[0, 1]

    model = BertLabeling.load_from_checkpoint(
        checkpoint_path=ckpt,
        hparams_file=hparams_file,
        map_location=None,
        batch_size=1,
        max_length=128,
        workers=0
    )
    trainer.test(model=model)


if __name__ == '__main__':
    # zh_msra
    #CHECKPOINTS = "/data/ai_group/mrc-ner/train_logs/zh_msra/zh_msra_bertlarge_lr8e-620200913_dropout0.2_bsz16_maxlen128/epoch=19_v0.ckpt"
    #HPARAMS = "/data/ai_group/mrc-ner/train_logs/zh_msra/zh_msra_bertlarge_lr8e-620200913_dropout0.2_bsz16_maxlen128/lightning_logs/version_0/hparams.yaml"

    #company
    CHECKPOINTS = "/root/mao/249/mrc-ner-company/train_logs/zh_msra_company/zh_msra_bertlarge_lr8e-620200913_dropout0.2_bsz16_maxlen128/epoch=19_v0.ckpt"
    HPARAMS = "/root/mao/249/mrc-ner-company/train_logs/zh_msra_company/zh_msra_bertlarge_lr8e-620200913_dropout0.2_bsz16_maxlen128/lightning_logs/version_0/hparams.yaml"



    evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)
