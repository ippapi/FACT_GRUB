
import os
import torch
import numpy as np
import random
from torch.optim import AdamW, Adam
import logging
from torch.utils.tensorboard import SummaryWriter
from transformers import  AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

logger = logging.getLogger("__main__")

def set_seed(seed, n_gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def get_optimizer(optimizer, model: torch.nn.Module, weight_decay: float = 0.0, lr: float = 0, adam_epsilon=1e-8) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
        'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if optimizer == "adamW":
        return AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    else:
        raise Exception("optimizer {0} not recognized! Can only be lamb or adamW".format(optimizer))


def set_env(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning("Device: %s", device)

    set_seed(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)

    basic_format = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
    formatter = logging.Formatter(basic_format)
    handler = logging.FileHandler(args.log_file, 'a', 'utf-8')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # logger.addHandler(sh)
    logger.setLevel(logging.INFO)
    print(logger)
    
def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(f'{args.model_name_or_path}')
    model = AutoModelForSequenceClassification.from_pretrained(f'{args.model_name_or_path}', num_labels=args.num_label) # 2 classification, support, refute, 
    print(f'Initialize {model.__class__.__name__} with parameters from {args.model_name_or_path}.')
    model.to(args.device)
    return tokenizer, model