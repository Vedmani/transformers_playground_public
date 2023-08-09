from datasets import load_dataset, disable_progress_bar, Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def get_imdb():
    dataset = load_dataset('imdb', split='train+test')
    return dataset


def get_train_val_dataset(dataset=get_imdb()):
    df = dataset.to_pandas()
    train_df, val_df = train_test_split(df, random_state=42, test_size=0.2, shuffle=True, stratify=df['label'])
    dataset_processed = DatasetDict({
        'train': Dataset.from_pandas(train_df, preserve_index=False),
        'validation': Dataset.from_pandas(val_df, preserve_index=False)
    })
    return dataset_processed