import torch
from torch.utils.data import DataLoader, SequentialSampler
import csv
import pandas as pd
from tqdm import tqdm
from utils.dataset import VerifierDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def predict(model, tokenizer, args, output_file="predictions.csv"):
    df_test = pd.read_csv(args.test_file)
    has_index = "index" in df_test.columns
    if has_index:
        test_indices = df_test["index"].tolist()
    else:
        test_indices = None
    
    test_dataset = VerifierDataset(
        args.test_file, 
        tokenizer=tokenizer, 
        max_len=args.max_len, 
        label_col=args.label_col,
        num_label=args.num_label
    )
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=test_dataset.create_mini_batch
    )

    model.to(args.device)
    model.eval()

    all_preds = []
    all_labels = []

    for batch in tqdm(test_dataloader, desc="Predicting"):
        batch = {k: v.to(args.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)

            if 'labels' in batch:
                all_labels.extend(batch['labels'].cpu().tolist())

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if has_index:
            if len(test_indices) != len(all_preds):
                raise ValueError("Số lượng index và số lượng dự đoán không khớp!")
            writer.writerow(["index", "prediction"])
            for idx, pred in zip(test_indices, all_preds):
                writer.writerow([idx, pred])
        else:
            writer.writerow(["prediction"])
            for pred in all_preds:
                writer.writerow([pred])

    print(f"Predictions saved to {output_file}")

    if all_labels:
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-score: {f1:.4f}")

    if has_index:
        return list(zip(test_indices, all_preds))
    else:
        return all_preds
