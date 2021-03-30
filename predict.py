# encoding: utf-8


import os
# from pytorch_lightning import Trainer

import torch
from trainer import BertLabeling
from datasets.mrc_ner_dataset_demo import MRCNERDataset_demo
from tokenizers import BertWordPieceTokenizer
from datasets.truncate_dataset import TruncateDataset
from torch.utils.data import DataLoader
from datasets.collate_functions import collate_to_max_length_demo


data_dir="datasets/zh_msra"
bert_dir="chinese_roberta_wwm_large_ext_pytorch"
max_length=128
batch_size=4
workers=0


def get_dataloader(prefix="train", limit: int = None) -> DataLoader:
    """get training dataloader"""
    """
    load_mmap_dataset
    """
    json_path = os.path.join(data_dir, f"mrc-ner.{prefix}")   #################### mrc-ner
    vocab_path = os.path.join(bert_dir, "vocab.txt")
    dataset = MRCNERDataset_demo(json_path=json_path,
                            tokenizer=BertWordPieceTokenizer(vocab_file=vocab_path),
                            max_length=max_length,
                            is_chinese=True,
                            pad_to_maxlen=False
                            )

    if limit is not None:
        dataset = TruncateDataset(dataset, limit)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True if prefix == "train" else False,
        collate_fn=collate_to_max_length_demo
    )

    return dataloader



if __name__ == '__main__':

    #company
    CHECKPOINTS = "/root/mao/249/mrc-ner-company/train_logs/zh_msra_company/zh_msra_bertlarge_lr8e-620200913_dropout0.2_bsz16_maxlen128/epoch=19_v0.ckpt"
    HPARAMS = "/root/mao/249/mrc-ner-company/train_logs/zh_msra_company/zh_msra_bertlarge_lr8e-620200913_dropout0.2_bsz16_maxlen128/lightning_logs/version_0/hparams.yaml"

    model = BertLabeling.load_from_checkpoint(
        checkpoint_path=CHECKPOINTS,
        hparams_file=HPARAMS,
        map_location=None,
        batch_size=1,
        max_length=128,
        workers=0
    )


    dataloader=get_dataloader('demo')

    vocab_file = os.path.join(bert_dir, "vocab.txt")
    tokenizer = BertWordPieceTokenizer(vocab_file=vocab_file)

    query = '公司指企业的组织形式，社会经济组织'

    with torch.no_grad():
        for batch in dataloader:          
            tokens, token_type_ids = batch

            attention_mask = (tokens != 0).long()
            start_logits, end_logits, span_logits = model(tokens, attention_mask, token_type_ids)

            ls_start=start_logits.squeeze().cpu().numpy().tolist()
            ls_end=end_logits.squeeze().cpu().numpy().tolist()

            for s, e, t in zip(ls_start, ls_end, tokens):
                ss=[i for i,v in enumerate(s) if v>0]
                ee=[i for i,v in enumerate(e) if v>0]

                t=t.tolist()
                t_d=tokenizer.decode(t, skip_special_tokens=True)
                print('\n', t_d[len(query)*2:])

                # print(ss, ee)

                if len(ss)==len(ee) and len(ss)>0:
                    for i, j in zip(ss, ee):
                        print('【Company】: ', tokenizer.decode(t[i:j+1]))
                else:
                    print('【Company】: None')



                

                