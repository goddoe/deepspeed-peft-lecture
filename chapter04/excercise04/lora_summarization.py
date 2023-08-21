from dataclasses import dataclass, field
from typing import Optional
import logging
import math
import sys
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    TaskType,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    LoraConfig
)

import evaluate
from datasets import load_dataset, load_from_disk
from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from tqdm import tqdm


logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    max_length: Optional[int] = field(
            default=None,
            metadata={
                "help": "max input length of model"
                }
            )


@dataclass
class PeftLoraArguments(LoraConfig):
    pass


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )


class PeftTrainer(Trainer):
    def _save_checkpoint(self, _, trial, metrics=None):
        """ Don't save base model, optimizer etc.
            but create checkpoint folder (needed for saving adapter) """
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (self.state.best_metric is None or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)):
                self.state.best_metric = metric_value

                self.state.best_model_checkpoint = output_dir

        os.makedirs(output_dir, exist_ok=True)

        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

class PeftSavingCallback(TrainerCallback):
    """ Correctly save PEFT model and not full model """

    def on_train_end(self, args: TrainingArguments, state: TrainerState,
            control: TrainerControl, **kwargs):
        """ Save final best model adapter """
        kwargs['model'].save_pretrained(args.output_dir)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState,
            control: TrainerControl, **kwargs):
        """ Save intermediate model adapters in case of interrupted training """
        save_path = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        kwargs['model'].save_pretrained(save_path)


def main():
    parser = HfArgumentParser((ModelArguments,
                               PeftLoraArguments,
                               DataTrainingArguments,
                               TrainingArguments))
    ######################################################################
    # Argparser
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, peft_lora_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, peft_lora_args, data_args, training_args = parser.parse_args_into_dataclasses()


    ######################################################################
    # Dataset
    dataset = load_dataset(data_args.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Helpers for dataset processing
    def build_prompted_example(title, document, summary):
        prompted_example = f"다음의 문서를 요약하시오\n제목: {title}\n본문: {document}\n요약: {summary}"
        return prompted_example 

    def tokenize_function(examples):
        prompted_examples = [build_prompted_example(title, document, summary)
                             for title, document, summary in zip(examples["title"],
                                                                 examples["document"],
                                                                 examples["summary"])]
        outputs = tokenizer(prompted_examples, truncation=True, max_length=model_args.max_length)
        return outputs

    def collate_fn(examples):
        examples_batch = tokenizer.pad(examples, padding="longest", return_tensors="pt")
        examples_batch['labels'] = examples_batch['input_ids']
        return examples_batch

    train_dataset = dataset['train']
    eval_dataset = dataset['validation']

    max_train_samples = len(train_dataset)
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    max_eval_samples = len(eval_dataset)
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    train_dataset_tokenized = train_dataset.map(tokenize_function,
                                                batched=True,
                                                remove_columns=['date',
                                                                'category',
                                                                'press',
                                                                'title',
                                                                'document',
                                                                'link',
                                                                'summary'])


    eval_data_tokenized = eval_dataset.map(tokenize_function,
                                           batched=True,
                                           remove_columns=['date',
                                                           'category',
                                                           'press',
                                                           'title',
                                                           'document',
                                                           'link',
                                                           'summary'])


    ######################################################################
    # Prepare Model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, return_dict=True)
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                             inference_mode=False,
                             r=peft_lora_args.r,
                             lora_alpha=peft_lora_args.lora_alpha,
                             lora_dropout=peft_lora_args.lora_dropout)

    model = get_peft_model(model, peft_config)
    logger.info(model.print_trainable_parameters())


    ######################################################################
    # Prepare Callbacked for Trainer
    
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)


    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)


    ######################################################################
    # Training
    trainer = PeftTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=eval_data_tokenized,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[PeftSavingCallback],
    )
    train_result = trainer.train()

    # trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    # trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__=='__main__':
    main()


