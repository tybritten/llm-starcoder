import logging
import os
import time
from argparse import Namespace
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.utils import ProjectConfiguration
from arguments import TrainingArguments
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
from attrdict import AttrDict

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, get_scheduler, set_seed


import determined as det
import json


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            tokenized (bool): If true we use a pretokenized dataset.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        tokenized=False,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.bos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.epoch = 0
        self.infinite = infinite
        self.current_size = 0
        self.tokenized = tokenized

        if self.tokenized:
            self.max_buffer_size = seq_length * num_of_sequences
            self.content_field = "input_ids"
        else:
            self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
            self.content_field = "content"

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        self.epoch += 1
                        logger.info(f"Dataset epoch: {self.epoch}")
                    else:
                        more_examples = False
                        break
            if self.tokenized:
                tokenized_inputs = buffer
            else:
                tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    self.current_size += 1
                    yield torch.tensor(input_ids)

    def shuffle(self, buffer_size=1000):
        return ShufflerIterDataPipe(self, buffer_size=buffer_size)


def setup_logging(args):
    logger = logging.getLogger(__name__)
    if accelerator.is_main_process:  # we only want to setup logging once
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_info()
        transformers.utils.logging.set_verbosity_info()
    else:
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    return logger


def create_dataloaders(args):
    ds_kwargs = {"streaming": True}
    train_data = load_dataset(args.dataset_name_train, split="train", **ds_kwargs)
    train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    valid_data = load_dataset(args.dataset_name_valid, split="train", **ds_kwargs)
    train_dataset = ConstantLengthDataset(
        tokenizer, train_data, infinite=True, seq_length=args.seq_length, tokenized=args.tokenized
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer, valid_data, infinite=False, seq_length=args.seq_length, tokenized=args.tokenized
    )
    train_dataset = train_dataset.shuffle(buffer_size=args.shuffle_buffer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    eval_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    return train_dataloader, eval_dataloader


def get_grouped_params(model, args, no_decay=["bias", "ln_1.weight", "ln_2.weight", "ln_f.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": args.weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def log_metrics(step, metrics):
    logger.info(f"Step {step}: {metrics}")
    if accelerator.is_main_process:
        accelerator.log(metrics, step)


def compute_tflops(elapsed_time, accelerator, args):
    # TFLOPs formula (from Equation 3 in Section 5.1 of https://arxiv.org/pdf/2104.04473.pdf).
    config_model = accelerator.unwrap_model(model).config
    checkpoint_factor = 4 if args.gradient_checkpointing else 3
    batch_size = args.train_batch_size * accelerator.state.num_processes * args.gradient_accumulation_steps
    factor = 24 * checkpoint_factor * batch_size * args.seq_length * config_model.n_layer * (config_model.n_embd**2)
    flops_per_iteration = factor * (
        1.0
        + (args.seq_length / (6.0 * config_model.n_embd))
        + (tokenizer.vocab_size / (16.0 * config_model.n_layer * config_model.n_embd))
    )
    tflops = flops_per_iteration / (elapsed_time * accelerator.state.num_processes * (10**12))
    return tflops


def evaluate(args):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(args.valid_batch_size)
        losses.append(accelerator.gather(loss))
        if args.max_eval_steps > 0 and step >= args.max_eval_steps:
            break
    losses = torch.cat(losses)
    loss = losses[: eval_dataloader.dataset.current_size].mean()
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()


# MLDE Integration
info = det.get_cluster_info()
assert info
hparams = info.trial.hparams

distributed = det.core.DistributedContext.from_torch_distributed()


# Settings
parser = HfArgumentParser(TrainingArguments)
args = parser.parse_args()

training_arguments = AttrDict(hparams.get("training_arguments", {}))
    
# MLDE set args from hyper parameters
for key in training_arguments:
    setattr(args, key, training_arguments[key])


with det.core.init(distributed=distributed) as core_context:

    for op in core_context.searcher.operations():
        args.max_train_steps = op.length

        # Accelerator
        accelerator = Accelerator(mixed_precision='fp16', dynamo_backend="INDUCTOR")
        acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items()}

        args = Namespace(**vars(args), **acc_state)
        samples_per_step = accelerator.state.num_processes * args.train_batch_size
        set_seed(args.seed)

        # Logging
        logger = setup_logging(args)
        logger.info(accelerator.state)


        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

        # Load dataset and dataloader
        train_dataloader, eval_dataloader = create_dataloaders(args)

        # Prepare the optimizer and learning rate scheduler
        optimizer = AdamW(get_grouped_params(model, args), lr=args.learning_rate)
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
        accelerator.register_for_checkpointing(lr_scheduler)


        def get_lr():
            return optimizer.param_groups[0]["lr"]


        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        #MLDE checkpoint logic
        if info.latest_checkpoint != None:
            latest_checkpoint = info.latest_checkpoint
            with core_context.checkpoint.restore_path(latest_checkpoint) as path:
                with open(path.joinpath("metadata.json"), "r") as f:
                    checkpoint_metadata_dict = json.load(f)
                    steps_completed = checkpoint_metadata_dict["steps_completed"]
                    ckpt_step_folder = f"step_{steps_completed}"
                    args.resume_from_checkpoint = path.joinpath(ckpt_step_folder)

        # load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
                accelerator.load_state(args.resume_from_checkpoint)
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(args.save_dir) if f.is_dir() and "step" in str(f)]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract the step of the checkpoint to continue from there
            training_difference = os.path.splitext(path)[0]
            resume_step = int(training_difference.replace("step_", ""))

        # Train model
        model.train()
        completed_steps = 0
        t_start = time.time()
        loss_tracking = 0
        for step, batch in enumerate(train_dataloader, start=1):
            if args.resume_from_checkpoint and step < resume_step:
                continue  # we need to skip steps until we reach the resumed step
            loss = model(batch, labels=batch, use_cache=False).loss
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            loss_tracking += avg_loss.item() / args.gradient_accumulation_steps
            if accelerator.is_main_process:
                core_context.train.report_training_metrics(steps_completed=step, metrics={"samples": step * samples_per_step, "loss_per_step/train": loss.item()})
                op.report_progress(step)
            loss = loss / args.gradient_accumulation_steps
            if step % args.gradient_accumulation_steps != 0:
                # Prevent backward from doing gradient all_reduce in every step
                if accelerator.distributed_type == DistributedType.MULTI_GPU:
                    with model.no_sync():
                        accelerator.backward(loss)
                else:
                    accelerator.backward(loss)
            else:
                lr = get_lr()
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                elapsed_time = time.time() - t_start
                tflops = compute_tflops(elapsed_time, accelerator, args)
                if accelerator.is_main_process:
                    core_context.train.report_training_metrics(
                        steps_completed=step,
                        metrics=
                        {
                            "steps": completed_steps,
                            "loss/train": loss_tracking,
                            "lr": lr,
                            "tflops": tflops,
                            "time_per_iteration": elapsed_time,
                        },
                    )
                    op.report_progress(step)
                t_start = time.time()
                loss_tracking = 0
                completed_steps += 1
            if step % args.save_checkpoint_steps == 0:
                logger.info("Evaluating and saving model checkpoint")
                eval_loss, perplexity = evaluate(args)                
                if accelerator.is_main_process:
                    core_context.train.report_validation_metrics(
                        steps_completed=step,
                        metrics={"loss/eval": eval_loss, "perplexity": perplexity}
                    )
                accelerator.wait_for_everyone()
                #MLDE save checkpoint - don't use local
                if accelerator.is_main_process:
                    with core_context.checkpoint.store_path({"steps_completed": step}) as (path, uuid):
                        save_dir = os.path.join(path, f"step_{step}")
                        print(f'saving checkpoint at step {step} to location {save_dir}')
                        accelerator.save_state(save_dir)
                        unwrapped_model = accelerator.unwrap_model(model)
                        print(f'saving pretrained at step {step} to location {save_dir}')
                        unwrapped_model.save_pretrained(save_dir, save_function=accelerator.save)

                if core_context.preempt.should_preempt():
                    raise Exception(f'Process preempted / killed at step {step}')
                model.train()
            if completed_steps >= args.max_train_steps:
                if accelerator.is_main_process:
                    op.report_completed(loss_tracking)
                break