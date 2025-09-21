import torch
from torch.nn.utils.rnn import pad_sequence

def fv_collate_fn(samples, tokenizer):
    input_ids = [s['input_ids'] for s in samples]
    attention_mask = [s['attention_mask'] for s in samples]
    labels = [s['label'] for s in samples]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    labels = torch.tensor(labels, dtype=torch.long)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

    return batch
