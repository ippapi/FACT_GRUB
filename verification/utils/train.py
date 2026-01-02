import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import logging
from tqdm import tqdm 
from tensorboardX import SummaryWriter
from utils.helper import get_optimizer
from utils.dataset import VerifierDataset
from utils.evaluate import evaluate_dev

logger = logging.getLogger("__main__")

def train(model, tokenizer, args):
    tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)

    optimizer = get_optimizer(
        args.optimizer, model,
        weight_decay=args.weight_decay,
        lr=args.lr,
        adam_epsilon=args.adam_epsilon
    )

    model.to(args.device)
    model.train()

    train_dataset = VerifierDataset(
        args.train_file, 
        tokenizer=tokenizer, 
        label_col="labels", 
        num_label=args.num_label
    )
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler,
        collate_fn=train_dataset.create_mini_batch,
        batch_size=args.per_device_train_batch_size
    )

    dev_dataloader = None
    best_dev_loss = float("inf")
    if args.validation_file is not None:
        dev_dataset = VerifierDataset(
            args.validation_file, 
            tokenizer=tokenizer,
            label_col="labels", 
            num_label=args.num_label
        )
        dev_sampler = SequentialSampler(dev_dataset)
        dev_dataloader = DataLoader(
            dev_dataset, sampler=dev_sampler,
            collate_fn=dev_dataset.create_mini_batch,
            batch_size=args.per_device_eval_batch_size
        )
        print(f"Loaded {len(dev_dataset)} dev samples.")

    trigger_times = 0

    for epoch in range(args.num_train_epochs):
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False)
        epoch_loss_sum = 0.0

        for batch in epoch_iterator:
            for k, v in batch.items():
                batch[k] = v.to(args.device)

            outputs = model(**batch)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss_sum += loss.item()

        avg_train_loss = epoch_loss_sum / len(train_dataloader)
        logger.info(f"Epoch {epoch+1} finished. Avg train_loss = {avg_train_loss:.4f}")
        tb_writer.add_scalar("avg_train_loss", avg_train_loss, epoch + 1)

        if dev_dataloader is not None:
            dev_results = evaluate_dev(model, dev_dataloader, args.device, num_labels=args.num_label)
            dev_loss = dev_results['average_loss']
            tb_writer.add_scalar("dev_loss", dev_loss, epoch + 1)
            logger.info(f"Epoch {epoch+1}: dev_loss = {dev_loss:.4f}, acc = {dev_results.get('accuracy', 'N/A')}")

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                trigger_times = 0
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                logger.info(f"Saved best model at end of epoch {epoch+1} with dev_loss={dev_loss:.4f}")
            else:
                trigger_times += 1
                logger.info(f"Trigger times: {trigger_times}")
                if trigger_times >= args.patience:
                    logger.info("Early stopping!")
                    tb_writer.close()
                    return best_dev_loss

    if dev_dataloader is None:
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Saved final model at {args.output_dir}")

    tb_writer.close()
    return best_dev_loss