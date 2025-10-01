import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
import os
import time
import argparse
import random
from tqdm import tqdm
import logging
import sys
from checker.utils.fv_dataset import FVDataset
from checker.verifier.evaluate import evaluate, evaluate_dev
from checker.utils.helper import get_optimizer, load_model, set_env, fv_collate_fn, levenshtein_filter

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger("__main__")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def classifier_filter(args, model, dataloader):
    datasize = len(dataloader.dataset)
    model.eval()
    logit_list = []
    prob_list = []
    idx_list = []
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluate"):
            for k, v in data.items():
                if k != 'idx_list':
                    data[k] = v.to(args.device)

            cur_idx = data['idx_list']
            if isinstance(cur_idx, torch.Tensor):
                cur_idx = cur_idx.tolist()
            elif not isinstance(cur_idx, list):
                cur_idx = [cur_idx]

            idx_list += cur_idx
            del data["idx_list"]

            output = model(**data)
            logits = output.logits
            probs = torch.softmax(logits, dim=-1)

            prob_list += probs.tolist()
            logit_list += logits.tolist()

    assert len(np.unique(idx_list)) == len(idx_list) == len(logit_list)

    filtered_idx_list = []    
    for idx, prob in zip(idx_list, prob_list):
        if prob[2] < args.cls_threshold and max(prob[0], prob[1]) > args.min_prob:
            filtered_idx_list.append(idx)

    return filtered_idx_list


def get_parameter():
    parser = argparse.ArgumentParser(description="Factual Error Correction.")
    parser.add_argument('--input_file', type=str, default='', help='The input file which contains the generated data.')
    parser.add_argument('--output_file', type=str, default='', help='The output file which contains the validated data.')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--use_leven', action='store_true', help='whether use the levenstein filter.')
    parser.add_argument('--leven_threshold', type=float, default=1, help='Remove the data instance, if the levenshtein edit distance between the src text and the generated text exceeds the theshold.')

    parser.add_argument('--use_cls', action='store_true', help='whether use the claissifier filter.')
    parser.add_argument('--cls_threshold', type=float, default=1/3, help='Remove the data instance, if the probilitiy for the "NOT ENOUGH INFO" class exceeds the theshold.')

    parser.add_argument('--num_workers', type=int, default=5, help='The number of processes to use for the preprocessing.')
    parser.add_argument('--min_prob', type=float, default=1/3, help='Remove the data instance, if the max probilitiy for the "REFUTES" and "SUPPORTS" class less than the min_prob.')
    parser.add_argument('--model_name', type=str, default='roberta-base')
    parser.add_argument('--max_len', type=int, default=128, help='the max length of the text.')
    parser.add_argument('--random_state', type = int, default = 16)
    parser.add_argument("--output_dir", type=str, default=None, help="dir for model checkpoints, logs and generated text.")
    parser.add_argument('--tensorboard_dir', type=str, default="None", help="Tensorboard log dir.")
    args = parser.parse_args()
    
    assert args.use_leven + args.use_cls >0
    
    return args

def main():
    args = get_parameter()

    df = pd.read_csv(args.input_file)

    data_instance_list = df.to_dict(orient="records")
    
    if args.use_leven:
        filtered_data_instance_list = []
        num_filter = 0
        for line in data_instance_list:
            src_text = line['Statement']
            generated_text = line['mutated']
            if not levenshtein_filter(src_text, generated_text, args.leven_threshold):
                filtered_data_instance_list.append(line)
            else:
                num_filter += 1
        data_instance_list = filtered_data_instance_list
        logger.info(f"{num_filter} data instances are filtered and {len(data_instance_list)} are left by levenshtein_filter.")
        
    if args.use_cls:
        set_env(args)
        tokenizer, model = load_model(args)
        dataset = FVDataset(
            data_instance_list, 
            tokenizer, 
            max_len=args.max_len,
            claim_column = "mutated"
        )
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset, 
            sampler=sampler,
            collate_fn=fv_collate_fn,
            batch_size=args.batch_size, 
            num_workers=args.num_workers
        )
        
        filtered_idx_list  = classifier_filter(args, model, dataloader)

        filtered_data_instance_list = []
        num_filter = len(data_instance_list) - len(filtered_idx_list)
        for idx in filtered_idx_list:
            line = data_instance_list[idx]
            filtered_data_instance_list.append(line)

        data_instance_list = filtered_data_instance_list

        logger.info(f"{num_filter} data instances are filtered and {len(data_instance_list)} are left by classifier_filter.")

    df_out = pd.DataFrame(data_instance_list)
    df_out.to_csv(args.output_file, index=False, encoding="utf-8")

if __name__ == "__main__":
    main()