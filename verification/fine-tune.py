import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW, Adam
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from transformers import  AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
import os
import argparse
import logging
from utils.helper import load_model, set_env, set_seed
from utils.train import train
from utils.evaluate import evaluate
from utils.predict import predict

logger = logging.getLogger("__main__")

def get_parameter():
    parser = argparse.ArgumentParser(description="Verifier for VENCE")
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_eval', action='store_true', help='Whether to run eval on the dev set.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predictions on the test set.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_label', type=int, default=3, help='3 or 2 label.')
    
    parser.add_argument('--dataset', type=str, default='data', help='the path of fact verification data.')
    parser.add_argument('--train_file', type=str, default=None, help='The input training data file (a jsonlines).')
    parser.add_argument('--validation_file', type=str, default=None, help='An optional input evaluation data file to evaluate the metrics (accuracy) on a jsonlines file.')  
    parser.add_argument('--test_file', type=str, default=None, help='An optional input test data file to evaluate the metrics (accuracy) on a jsonlines file.')
    parser.add_argument('--label_col', type=str, default=None, help='An optional input test data file have label or not.') 
    
    parser.add_argument('--dataset_percent', type=float, default=1,
                        help='The percentage of data used to train the model.')
    parser.add_argument('--num_data_instance', type=int, default=-1,
                        help='The number of data instances used to train the model. -1 denotes using all data.')
    parser.add_argument('--model_name_or_path', type=str, default='',
                        help='Path to pretrained model or model identifier from huggingface.co/models')
        
    parser.add_argument('--max_len', type=int, default=512, help='the max length of the text.')
    parser.add_argument('--num_train_epochs', type=int, default=-1)
    parser.add_argument('--optimizer', type=str, default='adamW', 
                        help='The optimizer to use.')
    parser.add_argument('--lr', type=float, default=4e-5, help='The initial learning rate for training.')
    parser.add_argument('--weight_decay', type=float, default=0.0, 
                        help='Weight decay for AdamW if we apply some.')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, 
                    help='Epsilon for AdamW optimizer.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, 
                help='Max gradient norm.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help= "Number of updates steps to accumulate before performing a backward/update pass.")
    
    parser.add_argument('--patience', type=int, default=2,
                        help='If the performance of model on the validation does not improve for n times, '
                            'we will stop training.')
    parser.add_argument('--per_device_train_batch_size', type=int, default=64)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=128)
    
    parser.add_argument('--logging_steps', type=int, default=10, help='Log every X updates steps.')
    parser.add_argument('--save_steps', type=int, default = 10, help='Save checkpoint every X updates steps.')
    parser.add_argument('--tensorboard_dir', type=str, default="../tensorboard_log", help="Tensorboard log dir.")
    parser.add_argument("--output_dir", type=str, default=None, help="dir for model checkpoints, logs and generated text.")
    parser.add_argument("--filter_output", type=str, default=None, help="dir for filtered text")
    parser.add_argument('--resume', action='store_true', help='whether load the best checkpoint or not.')
    parser.add_argument('--device', type = str, default = 'cuda')
    
    args = parser.parse_args()
    assert args.do_train + args.do_eval + args.do_predict == 1, print('Specify do_train, do_eval or do_predict.')
    
    if args.resume:
        print(args.model_name_or_path)
        assert os.path.exists(args.model_name_or_path), print('Please provide the checkpoint.')
        args.output_dir = args.model_name_or_path
        args.tensorboard_dir = args.output_dir.replace("checkpoints_verifier", "tensorboard_log_verifier")
    else:
        dir_prefix = f"seed{args.seed}_lr{args.lr}"
        
        if args.dataset_percent<1:
            dir_prefix += f'_{args.dataset_percent}-data-percent'
        if args.num_data_instance>0:
            dir_prefix += f'_{args.num_data_instance}-data-instance'
        assert args.dataset_percent==1 or args.num_data_instance==-1, print("Do not set both dataset_percent and num_data_instance.")
        
        if args.output_dir is None:
            if args.do_train:
                args.output_dir = f'./train_verifier/{dir_prefix}'
            else:
                args.output_dir = args.model_name_or_path
        args.tensorboard_dir = f'./tensorboard_log_verifier/{dir_prefix}'
    args.log_file = f'{args.output_dir}/log.txt'
    
    
    return args

    
def main():
    
    args = get_parameter()
    set_env(args)
    tokenizer, model = load_model(args)
    
    if args.do_train:
        logger.info("*** Train ***")
        logger.info("args:\n%s", '\n'.join([f'    {arg}={getattr(args, arg)}'  for arg in vars(args)]))
        best_dev_loss = train(model, tokenizer, args)
        logger.info(" best_dev_loss = %s", best_dev_loss)
        
    if args.do_eval:
        logger.info("*** Evaluate ***") 
        print("args:\n%s", '\n'.join([f'    {arg}={getattr(args, arg)}'  for arg in vars(args)]))
        results = evaluate(model, tokenizer, args)
        for key, value in results.items():
            print(f"{key}: {value:.4f}")

    if args.do_predict:
        logger.info("*** Predict ***")
        
        output_dir = os.path.join(args.output_dir, "file")
        output_file = os.path.join(output_dir, "predictions-lime-sbert-retri.csv")
        os.makedirs(output_dir, exist_ok=True)
        
        predict(model, tokenizer, args, output_file)
        logger.info(f"Predictions saved to {output_file}")
        
        
if __name__ == "__main__":
    main()