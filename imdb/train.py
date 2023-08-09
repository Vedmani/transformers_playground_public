import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_PROJECT"] = "imdb"

import transformers
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry

import evaluate

from config import get_args
from model import get_cls_model, get_model_peft
from toknizer_utils import get_tokenizer, tokenize_dataset
from data_utils import get_train_val_dataset

model_name = "bert-base-cased"
tok = get_tokenizer(model_name)
data_collator = DataCollatorWithPadding(tokenizer=tok, max_length=None)

dataset = get_train_val_dataset()
print(dataset)

dataset = tokenize_dataset(dataset, tok)

model = get_cls_model(model_name_or_path='bert-base-cased', num_classes=2, seed=42)
model = get_model_peft(model, seed=42)

accuracy = evaluate.load("accuracy")
roc_auc_score = evaluate.load("roc_auc")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = (accuracy.compute(predictions=predictions, references=labels))['accuracy']
    roc_auc = (roc_auc_score.compute(predictions=predictions, references=labels))['roc_auc']
    return {'accuracy': acc, 'roc_auc': roc_auc}


args = get_args()

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    tokenizer=tok,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model()
model.save_pretrained(args.output_dir+"/save_model")
merged_model = model.merge_and_unload()
merged_model.save_pretrained(args.output_dir+"/merged_model")