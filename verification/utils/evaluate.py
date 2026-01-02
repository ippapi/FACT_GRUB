import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils.dataset import VerifierDataset
from sklearn.metrics import f1_score

def evaluate_dev(model, dataloader, device, num_labels=3):
    model.eval()
    total_loss, correct = 0.0, 0
    corrects = torch.zeros(num_labels, device=device)
    total_labels = torch.zeros(num_labels, device=device)
    total_preds = torch.zeros(num_labels, device=device)
    
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating ..."):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("token_type_ids", None)
            
            outputs = model(**batch)
            loss, logits = outputs['loss'], outputs['logits']
            preds = logits.argmax(dim=1)

            all_labels.extend(batch['labels'].cpu().tolist())
            all_predictions.extend(preds.cpu().tolist())

            correct += (preds == batch['labels']).sum().item()
            total_loss += loss.item() * batch['input_ids'].size(0)

            for i in range(num_labels):
                mask = (batch['labels'] == i)
                corrects[i] += (preds[mask] == i).sum()
                total_labels[i] += mask.sum()
                total_preds[i] += (preds == i).sum()

    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    results = {
        "average_loss": round(total_loss / len(dataloader.dataset), 4),
        "accuracy": round(correct / len(dataloader.dataset), 4),
        "f1_macro": round(f1_macro, 4)
    }

    model.train()
    return results

def evaluate(model, tokenizer, args):
    dev_dataset = VerifierDataset(
        args.validation_file, 
        tokenizer=tokenizer,
        label_col=args.label_col,
        num_label=args.num_label
    )
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(
        dev_dataset, sampler=dev_sampler,
        collate_fn=dev_dataset.create_mini_batch,
        batch_size=args.per_device_eval_batch_size
    )
    
    results = evaluate_dev(model, dev_dataloader, args.device, num_labels=args.num_label)
    return results