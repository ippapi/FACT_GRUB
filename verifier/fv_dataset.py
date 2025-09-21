import torch
from torch.utils.data import Dataset
import os, sys
import pandas as pd

LABEL_DICT = {"SUPPORTS":0, "REFUTES":1, "NOT ENOUGH INFO":2}

@contextmanager
def suppress_stdout():
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

class FVDataset(Dataset):
    """
        This class is used to handle fact verification task and can either:
        1. Tokenize input (claim <sep> evidence).
        2. Support load data from dataset or list, dicts.
    """

    def __init__(self, data_source, tokenizer = None, max_len = None, transform = None, claim_column = "claim", evidence_column = "evidence", label_column = "label"):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

        self.claim_column = claim_column
        self.evidence_column = evidence_column
        self.label_column = label_column
        self.label_map = LABEL_DICT
        
        if isinstance(data_source, str):
            df = pd.read_json(data_source) if data_source.endswith('.json') else pd.read_csv(data_source)
            self.data = df.to_dict(orient = 'records')
        else:
            self.data = data_source

        for item in self.data:
            raw_label = item[self.label_column]
            if raw_label in self.label_map:
                item[self.label_column] = self.label_map[raw_label]
            else:
                raise ValueError(f"Label {raw_label} không có trong label_map!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        claim = instance[self.claim_column]
        evidence = instance[self.evidence_column]

        if self.transform:
            evidence = self.transform(evidence)

        with suppress_stdout():
            inputs = self.tokenizer(
                claim,
                evidence,
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                add_special_tokens=True,
                return_tensors='pt'
            )

        item = {
            'input_ids': inputs['input_ids'].flatten(),        # flatten thay vì squeeze
            'attention_mask': inputs['attention_mask'].flatten(),
            'label': torch.tensor(instance[self.label_column], dtype=torch.long),
            'idx': idx
        }

        return item

