import os
import numpy as np
import random
import torch
import Levenshtein
from torch.optim import AdamW, Adam
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def fv_collate_fn(samples):
    input_ids = torch.stack([s['input_ids'] for s in samples])
    attention_mask = torch.stack([s['attention_mask'] for s in samples])
    labels = torch.stack([s['label'] for s in samples])
    idx_list = [s['idx'] for s in samples] 

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'idx_list': idx_list
    }

def set_env(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.random_state:
        set_seed(args.random_state)
    else:
        set_seed(16)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.tensorboard_dir:
        os.makedirs(args.tensorboard_dir, exist_ok=True)


def get_optimizer(opt_name, model, lr, weight_decay=0.0, adam_epsilon=1e-8):
    if opt_name.lower() == "adamw":
        return AdamW(model.parameters(), lr=lr, eps=adam_epsilon, weight_decay=weight_decay)
    elif opt_name.lower() == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

def load_model(args):    
    tokenizer = AutoTokenizer.from_pretrained(f'{args.model_name}')
    model = AutoModelForSequenceClassification.from_pretrained(f'{args.model_name}', num_labels=3)

    model.to(args.device)
    return tokenizer, model

def levenshtein_filter(src_text, generated_text, threshold):
    """
    Filter out the generated_text if the Levenshtein distance between it and the source text over the threshold. 
    """
    if src_text.strip() == generated_text.strip():
        return True
    if Levenshtein.distance(src_text, generated_text) / len(src_text) >threshold:
        return True 
    return False