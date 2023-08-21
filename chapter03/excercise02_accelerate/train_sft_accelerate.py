import os
from argparse import ArgumentParser
from tqdm.auto import tqdm

import torch
from torch.optim import AdamW
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset



def write_metric_dict(writer, metric_dict, global_i):
    for k, v in metric_dict.items():
        writer.add_scalar(k, v, global_i)

def main(args):
    
    ###############################################################################
    # Define Model
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path)
    
    
    # GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
    # only for this model.
    tokenizer.pad_token = tokenizer.eos_token
    
    
    def collate_fn(batch):
        return {k: torch.stack([torch.tensor(b[k], dtype=torch.long) for b in batch] ).long() for k in batch[0].keys()}
    
    
    dataset = load_dataset(args.dataset_name)
    
    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["text"], padding="max_length", max_length=args.max_length, truncation=True)
        return outputs
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["source", "text"])
    
    tmp_td = tokenized_dataset['train'].train_test_split(train_size=0.8)
    
    tokenized_dataset['train'] = tmp_td['train']
    tokenized_dataset['validation'] = tmp_td['test']
    
    train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(tokenized_dataset['validation'], batch_size=args.batch_size, collate_fn=collate_fn)
    
    
    def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
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
    
    
    def evaluate():
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(batch["input_ids"], labels=batch["input_ids"])
    
            losses.append(accelerator.gather(outputs.loss))
        loss = torch.mean(torch.stack(losses))
        try:
            perplexity = torch.exp(loss)
        except OverflowError:
            perplexity = float("inf")
        return loss.item(), perplexity.item()
    
    
    optimizer = AdamW(get_grouped_params(model), lr=args.lr)
    
    accelerator = Accelerator()
    
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = args.n_epoch * num_update_steps_per_epoch
    
    if args.warmpup_ratio:
        num_warmup_steps = int(args.n_epoch * args.warmpup_ratio)
    
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    (model,
     optimizer,
     train_dataloader,
     eval_dataloader,
     lr_scheduler) = accelerator.prepare(model,
                                         optimizer,
                                         train_dataloader,
                                         eval_dataloader,
                                         lr_scheduler)
    
    if args.tensorboard_path:
        tensorboard_path = args.tensorboard_path
    else:
        tensorboard_path = os.path.join(args.save_dir, 'tensorboard')
    writer = SummaryWriter(tensorboard_path)
    
    
    global_i = 0
    
    min_eval_loss = 9e+9
    
    model.train()
    completed_steps = 0
    
    
    for epoch in range(args.n_epoch):
        for step, batch in tqdm(
            enumerate(train_dataloader, start=1), total=num_training_steps
        ):

            x = batch["input_ids"]

            print(f"x: {x}")

            out = model(x, labels=x)
    
            loss = out.loss
            metric_dict = {"lr": lr_scheduler.get_lr()[0],
                           "steps": completed_steps,
                           "loss/train": loss.item() * args.gradient_accumulation_steps,
                           }
    
            if global_i % args.verbose_interval == 0:
                accelerator.print(metric_dict)
    
            if global_i % args.tensorboard_log_interval == 0 and accelerator.is_main_process:
                write_metric_dict(writer, metric_dict, completed_steps)
    
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
    
            if (global_i % (args.save_interval * args.gradient_accumulation_steps)) == 0:
                eval_loss, perplexity = evaluate()
                metric_dict = {"loss/eval": eval_loss, "perplexity": perplexity}
                accelerator.print(metric_dict)
                write_metric_dict(writer, metric_dict, completed_steps)
    
                model.train()
                accelerator.wait_for_everyone()
    
                output_path = os.path.join(args.save_dir, f"model_{completed_steps}")
    
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(output_path,
                                                is_main_process=accelerator.is_main_process,
                                                save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(output_path)
    
                if eval_loss < min_eval_loss:
                    min_eval_loss = eval_loss
    
                    best_path = os.path.join(args.save_dir, "model_best")
                    unwrapped_model.save_pretrained(best_path,
                                                    is_main_process=accelerator.is_main_process,
                                                    save_function=accelerator.save)
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(best_path)
    
            global_i += 1
    

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default="EleutherAI/polyglot-ko-1.3b")
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--dataset_name", type=str, default="heegyu/korquad-chat-v1")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--n_epoch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmpup_ratio", type=float, default=0.06)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--verbose_interval", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--tensorboard_log_interval", type=int, default=20)
    parser.add_argument("--tensorboard_path", type=str, default="./tensorboard")

    args = parser.parse_args()
    set_seed(args.seed)

    main(args)



