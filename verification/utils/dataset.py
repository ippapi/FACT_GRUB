from torch.utils.data import Dataset
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import logging
logging.set_verbosity_error() 

valid_labels = {0, 1, 2}

class VerifierDataset(Dataset):
    def __init__(self, filename, tokenizer=None, max_len=256, 
                 claim_col="Statement", evidence_col="Evidence", label_col="labels", num_label=3):
        self.filename = filename
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.claim_col = claim_col
        self.evidence_col = evidence_col
        self.label_col = label_col
        self.num_label = num_label
        
        df = pd.read_parquet(filename) if filename.endswith('.parquet') else pd.read_csv(filename)

        if label_col is None:
            self.has_label = False
        else:
            self.has_label = label_col in df.columns
        
        if self.has_label and self.num_label is not None:
            valid_labels = set(range(self.num_label))
            df = df[df[label_col].isin(valid_labels)].reset_index(drop=True)
            df[label_col] = df[label_col].astype(int)

        self.data_list = []
        for _, row in df.iterrows():
            item = {
                "claim": row[claim_col],
                "evidence": row[evidence_col],
            }
            if self.has_label:
                item["label"] = row[label_col]
            self.data_list.append(item)
            
        self.len = len(self.data_list)
        print(f"Loaded {len(self.data_list)} valid samples.")
        
    def __getitem__(self, idx):
        data_instance = self.data_list[idx]

        inputs = self.tokenizer(
            data_instance['claim'], data_instance['evidence'], 
            max_length=self.max_len,
            truncation='longest_first',
            padding='max_length',
            add_special_tokens=True,
            return_tensors='pt'
        )
        result = {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "token_type_ids": inputs["token_type_ids"][0] if "token_type_ids" in inputs else None,
            "idx": idx,
        }

        if self.has_label:
            result["label"] = data_instance["label"]

        return result
        
    def __len__(self):
        return self.len
    

    def create_mini_batch(self, samples):
        input_ids = [s['input_ids'] for s in samples]
        attention_mask = [s['attention_mask'] for s in samples]
    
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
        if samples[0]['token_type_ids'] is None:
            token_type_ids = None
        else:
            token_type_ids = [s['token_type_ids'] for s in samples]
            token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.has_label:
            labels = torch.tensor([s["label"] for s in samples], dtype=torch.long)
            batch["labels"] = labels
            
        if token_type_ids is not None:
            batch["token_type_ids"] = token_type_ids
    
        return batch