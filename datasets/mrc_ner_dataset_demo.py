# encoding: utf-8


import json
import torch
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset


class MRCNERDataset_demo(Dataset):
    """
    MRC NER Dataset
    Args:
        json_path: path to mrc-ner style json
        tokenizer: BertTokenizer
        max_length: int, max length of query+context
        possible_only: if True, only use possible samples that contain answer for the query/context
        is_chinese: is chinese dataset
    """
    def __init__(self, json_path, tokenizer: BertWordPieceTokenizer, max_length: int = 128, possible_only=False,
                 is_chinese=False, pad_to_maxlen=False):
        # self.all_data = json.load(open(json_path, encoding="utf-8"))
        f=open(json_path, 'r', encoding='utf-8')
        self.all_data = f.read().strip().split('\n')
        f.close()

        self.tokenzier = tokenizer
        self.max_length = max_length
        self.possible_only = possible_only
        # if self.possible_only:
        #     self.all_data = [
        #         x for x in self.all_data if x["start_position"]
        #     ]
        self.is_chinese = is_chinese
        self.pad_to_maxlen = pad_to_maxlen

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of query + context, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labelsof NER in tokens, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
            match_labels: match labels, [seq_len, seq_len]
            sample_idx: sample id
            label_idx: label id

        """
        data = self.all_data[item]
        tokenizer = self.tokenzier

        # qas_id = data.get("qas_id", "0.0") ### 第二个参数是默认返回值
        # sample_idx, label_idx = qas_id.split(".")
        # sample_idx = torch.LongTensor([int(sample_idx)])
        # label_idx = torch.LongTensor([int(label_idx)])

        query = '公司指企业的组织形式，社会经济组织'
        context = data

        query_context_tokens = tokenizer.encode(query, context, add_special_tokens=True)
        tokens = query_context_tokens.ids
        type_ids = query_context_tokens.type_ids

        # truncate
        tokens = tokens[: self.max_length]
        type_ids = type_ids[: self.max_length]

        # make sure last token is [SEP]
        sep_token = tokenizer.token_to_id("[SEP]")
        if tokens[-1] != sep_token:
            assert len(tokens) == self.max_length
            tokens = tokens[: -1] + [sep_token]

        if self.pad_to_maxlen:
            tokens = self.pad(tokens, 0)
            type_ids = self.pad(type_ids, 1)

        return [
            torch.LongTensor(tokens),
            torch.LongTensor(type_ids)
            # torch.LongTensor(start_labels),
            # torch.LongTensor(end_labels),
            # torch.LongTensor(start_label_mask),
            # torch.LongTensor(end_label_mask),
            # match_labels,
            # sample_idx,
            # label_idx
        ]

    def pad(self, lst, value=0, max_length=None):
        max_length = max_length or self.max_length
        while len(lst) < max_length:
            lst.append(value)
        return lst


def run_dataset():
    """test dataset"""
    import os

    #from datasets.collate_functions import collate_to_max_length
    from collate_functions import collate_to_max_length, collate_to_max_length_demo

    from torch.utils.data import DataLoader
    # zh datasets
    # bert_path = "/mnt/mrc/chinese_L-12_H-768_A-12"
    # json_path = "/mnt/mrc/zh_msra/mrc-ner.test"
    # # json_path = "/mnt/mrc/zh_onto4/mrc-ner.train"
    # is_chinese = True

    # en datasets
    bert_path = "../chinese_roberta_wwm_large_ext_pytorch"
    json_path = "zh_msra/mrc-ner.demo"
    # json_path = "/mnt/mrc/genia/mrc-ner.train"
    is_chinese = False

    vocab_file = os.path.join(bert_path, "vocab.txt")
    #assert os.path.exists(vocab_file)
    tokenizer = BertWordPieceTokenizer(vocab_file=vocab_file)

    dataset = MRCNERDataset_demo(json_path=json_path, tokenizer=tokenizer,
                            is_chinese=is_chinese)

    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_to_max_length_demo)
    # dataloader = DataLoader(dataset, batch_size=32)

    for batch in dataloader:
        for tokens, token_type_ids in zip(*batch):
            tokens = tokens.tolist()

            print("="*20)
            print(f"len: {len(tokens)}", tokenizer.decode(tokens, skip_special_tokens=False))
            # for start, end in zip(start_positions, end_positions):
                # print(str(sample_idx.item()), str(label_idx.item()) + "\t" + tokenizer.decode(tokens[start: end+1]))


if __name__ == '__main__':
    run_dataset()
