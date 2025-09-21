import torch

def fv_collate_fn(samples):
    input_ids = torch.stack([s['input_ids'] for s in samples])
    attention_mask = torch.stack([s['attention_mask'] for s in samples])
    labels = torch.stack([s['label'] for s in samples])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


