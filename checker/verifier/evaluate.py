import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from checker.utils.fv_dataset import FVDataset
from checker.utils.helper import fv_collate_fn

def evaluate_dev(model, dataloader, device, num_labels=3):
    model.eval()
    total_loss = 0.0
    correct = 0

    corrects = torch.zeros(num_labels, device=device)
    total_labels = torch.zeros(num_labels, device=device)
    total_preds = torch.zeros(num_labels, device=device)

    datasize = len(dataloader.dataset)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Info] Evaluating ..."):
            batch = {k: v.to(device) for k, v in batch.items()}
            if "token_type_ids" in batch:
                batch.pop("token_type_ids")
            outputs = model(**batch)
            loss, logits = outputs['loss'], outputs['logits']
            preds = torch.argmax(logits, dim=1)

            correct += (preds == batch['labels']).sum().item()
            total_loss += loss.item() * batch['input_ids'].size(0)

            for i in range(num_labels):
                corrects[i] += ((preds == i) & (batch['labels'] == i)).sum()
                total_labels[i] += (batch['labels'] == i).sum()
                total_preds[i] += (preds == i).sum()

    recalls = (corrects / total_labels).cpu().tolist()
    precisions = (corrects / total_preds).cpu().tolist()
    f1s = [2*r*p/(r+p) if r+p > 0 else 0 for r, p in zip(recalls, precisions)]

    # 👉 Macro averages
    macro_recall = sum(recalls) / num_labels
    macro_precision = sum(precisions) / num_labels
    macro_f1 = sum(f1s) / num_labels

    average_loss = total_loss / datasize
    accuracy = correct / datasize

    results = {
        "average_loss": round(average_loss, 4),
        "accuracy": round(accuracy, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_recalls": [round(r, 4) for r in recalls],
        "per_class_precisions": [round(p, 4) for p in precisions],
        "per_class_f1s": [round(f, 4) for f in f1s]
    }

    print(results)

    model.train()
    return results

def evaluate(model, tokenizer, args):
    dev_dataset = FVDataset(
        args.dev_file, 
        tokenizer, 
        max_len=args.max_len, 
    )

    sampler = SequentialSampler(dev_dataset)

    dev_loader = DataLoader(
        dev_dataset, 
        sampler = sampler,
        collate_fn = fv_collate_fn,
        batch_size = args.batch_size, 
        num_workers = args.num_workers
    )

    results = evaluate_dev(model, dev_loader, args.device)

    return results
