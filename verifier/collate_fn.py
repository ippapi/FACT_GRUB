import torch

def fv_collate_fn(samples, tokenizer):
    input_ids_list = [s['input_ids'].tolist() for s in samples]
    attention_mask_list = [s['attention_mask'].tolist() for s in samples]
    labels = torch.tensor([s['label'] for s in samples], dtype=torch.long)

    batch_encoding = tokenizer.pad(
        {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list
        },
        padding=True,
        return_tensors="pt"
    )

    batch_encoding["labels"] = labels
    return batch_encoding

