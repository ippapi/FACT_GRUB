import os
import torch
from torch.optim import AdamW, Adam
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.tensorboard import SummaryWriter

def set_env(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
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