import torch
from torch.utils.data import Dataset
from log import logger
import os
import pandas as pd

class S2SDataset(Dataset):
    """
        Seq2seqDataset with caching support.

        This class is used to handle sequence to sequence task and can either:
        1. Tokenize source and target texts dynamically.
        2. Support load data from dataset or list, dicts.
    """

    def __init__(self, data_source, src_transform = None, tokenizer = None, max_src_len = None, max_tgt_len = None, src_column = "src", tgt_column = None):
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.src_column = src_column
        self.tgt_column = tgt_column
        self.src_transform = src_transform
        
        if isinstance(data_source, str):
            df = pd.read_json(data_source) if data_source.endswith('.json') else pd.read_csv(data_source)
            self.data = df.to_dict(orient = 'records')
        else:
            self.data = data_source

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]

        src = self.src_transform(instance)
        src_tokenization = torch.tensor(self.tokeniizer.encode(src, max_length=self.max_src_len, truncation=True, padding=False, add_special_tokens=True), dtype=torch.long)

        tgt_tokenization = None
        if self.tgt:
            tgt = data_instance[self.tgt]
            if 'T5Tokenizer' in self.tokenizer.__class__.__name__:
                tgt_tokenization = torch.tensor([self.tokenizer.pad_token_id] + self.tokenizer.encode(tgt, max_length=self.max_tgt_len, truncation=True, padding=False, add_special_tokens=True), dtype=torch.long)
            else:
                tgt_tokenization = torch.tensor(self.tokenizer.encode(tgt, max_length=self.max_tgt_len, truncation=True, padding=False, add_special_tokens=True), dtype=torch.long)

        return {
            "src_tokenization": src_tokenization,
            "tgt_tokenization": tgt_tokenization,
            "idx": idx
        }        

