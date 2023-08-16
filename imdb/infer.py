from model import get_cls_model
from peft import PeftModel, PeftConfig
from config import get_args
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

import evaluate

from config import get_args
from model import get_cls_model, get_model_peft
from toknizer_utils import get_tokenizer, tokenize_dataset
from data_utils import get_train_val_dataset

model_name = "bert-base-cased"
tok = get_tokenizer(model_name)
data_collator = DataCollatorWithPadding(tokenizer=tok, max_length=None)

training_args, custom_args = get_args()
model = get_cls_model(model_name_or_path='bert-base-cased', num_classes=2, seed=42)
model = PeftModel.from_pretrained(model, custom_args.model_save_path)


dataset = get_train_val_dataset()
print(dataset)

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    tokenizer=tok,
    compute_metrics=compute_metrics,
)

trainer.evaluate()