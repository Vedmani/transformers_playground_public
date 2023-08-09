from dataclasses import dataclass, field
from transformers import TrainingArguments, HfArgumentParser
import torch


@dataclass
class MyTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default='output_new',
        metadata={'help': 'The output directory where the model predictions and checkpoints will be written.'}
    )

    overwrite_output_dir: bool = field(
        default=True,
        metadata={'help': 'If True, overwrite the content of the output directory.'}
    )

    do_train: bool = field(
        default=True,
        metadata={'help': 'Whether to run training or not.'}
    )

    do_eval: bool = field(
        default=True,
        metadata={'help': 'Whether to run evaluation on the validation set or not.'}
    )

    evaluation_strategy: str = field(
        default='epoch',
        metadata={'help': 'The evaluation strategy to adopt during training.'}
    )

    per_device_train_batch_size: int = field(
        default=32,
        metadata={'help': 'The batch size per GPU/TPU core/CPU for training.'}
    )

    per_device_eval_batch_size: int = field(
        default=32,
        metadata={'help': 'The batch size per GPU/TPU core/CPU for evaluation.'}
    )

    learning_rate: float = field(
        default=5e-4,
        metadata={'help': 'The initial learning rate for Adam.'}
    )

    num_train_epochs: float = field(
        default=3.0,
        metadata={'help': 'Total number of training epochs to perform.'}
    )
    optim: str = field(
        default="adamw_torch",
        metadata={'help': 'The optimizer to use.'}
    )

    lr_scheduler_type: str = field(
        default='cosine',
        metadata={'help': 'The learning rate scheduler type.'}
    )

    log_level: str = field(
        default='error',
        metadata={'help': 'The log level to use.'}
    )
    logging_steps: int = field(
        default=25,
        metadata={'help': 'Log every X updates steps.'}
    )

    save_strategy: str = field(
        default='epoch', 
        metadata={'help': 'The checkpoint save strategy to adopt during training.'}
    )

    save_total_limit: int = field(
        default=2, 
        metadata={'help': 'Limit the total amount of checkpoints. Deletes the older checkpoints.'}
    )

    fp16: bool = field(
        default=torch.cuda.is_available(), 
        metadata={'help': 'Whether to use 16-bit (mixed) precision training or not.'}
    )

    metric_for_best_model: str = field(
        default='accuracy', 
        metadata={'help': 'The metric to use to compare models.'}
    )

    greater_is_better: bool = field(
        default=True, 
        metadata={'help': 'Whether a high metric value is better or not.'}
    )

    group_by_length: bool = field(
        default=True, 
        metadata={'help': 'Whether to group samples of similar length together.'}
    )

    report_to: str = field(
        default='wandb', 
        metadata={'help': 'The list of integrations to report the results and logs to.'}
    )

    dataloader_pin_memory: bool = field(
        default=True, 
        metadata={'help': 'Whether you want to pin memory in data loaders or not.'}
    )

    auto_find_batch_size: bool = field(
        default=False,
        metadata={'help': 'Whether to find a batch size that will fit into memory automatically.'}
    )

    load_best_model_at_end: bool = field(
        default=False, 
        metadata={'help': 'Whether to load the best model found during training at the end.'}
    )

    logging_strategy: str = field(
        default='steps',
        metadata={'help': 'The logging strategy to adopt during training.'}
    )

    logging_steps: int or float = field(
        default=50,
        metadata={'help': 'Log every X updates steps.'}
    )

    warmup_ratio: float = field(
        default=0.3,
        metadata={'help': 'Linear warmup over warmup_steps.'}
    )


def get_args():
    parser = HfArgumentParser(MyTrainingArguments)
    training_args, = parser.parse_args_into_dataclasses()
    return training_args
