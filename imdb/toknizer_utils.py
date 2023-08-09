from transformers import AutoTokenizer


def get_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name,
                                        use_fast=True,)
    return tok


def tokenize(example, tokenizer):
    return tokenizer(example['text'], add_special_tokens=True, truncation=True, max_length=None)


def tokenize_dataset(dataset, tokenizer):
    tokenized_dataset = dataset.map(tokenize, batched=True, fn_kwargs={'tokenizer': tokenizer}, remove_columns=['text'])
    return tokenized_dataset
