import torch
from torch.nn.utils.rnn import pad_sequence

def fv_collate_fn(samples, tokenizer):
    input_ids = [s['input_ids'] for s in samples]
    attention_mask = [s['attention_mask'] for s in samples]
    token_type_ids = [s.get('token_type_ids') for s in samples]
    labels = [s['label'] for s in samples]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

    if all(t is None for t in token_type_ids):
        token_type_ids = None
    else:
        fixed_token_type_ids = []
        for t, s in zip(token_type_ids, samples):
            if t is not None:
                fixed_token_type_ids.append(t)
            else:
                fixed_token_type_ids.append(torch.zeros_like(s['input_ids']))
        token_type_ids = pad_sequence(fixed_token_type_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    labels = torch.tensor(labels, dtype=torch.long)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    if token_type_ids is not None:
        batch["token_type_ids"] = token_type_ids

    return batch
