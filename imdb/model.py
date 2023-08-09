from transformers import AutoModel, AutoModelForSequenceClassification, Trainer, set_seed
import torch
from peft import get_peft_model
from peft import LoraConfig, TaskType


def get_cls_model(model_name_or_path, num_classes, seed=42):
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_classes)
    model.config.classifier_dropout = 0
    return model


def get_model_peft(model, task_type=TaskType.SEQ_CLS, seed=42):
    set_seed(seed)
    peft_config = LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=8, lora_alpha=32,
        lora_dropout=0.1, 
        modules_to_save=["classifier"])
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
