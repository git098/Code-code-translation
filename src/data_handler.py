import pandas as pd
from datasets import Dataset
import os

def load_code_pairs(java_file, cs_file):
    """Loads Java and C# code pairs from text files."""
    with open(java_file, 'r') as f:
        java_code = f.readlines()
    with open(cs_file, 'r') as f:
        cs_code = f.readlines()
    return pd.DataFrame({
        'java_code': java_code,
        'cs_code': cs_code
    })

def reduce_dataset_size(df, percentage=0.1):
    """Reduces the size of the dataset."""
    return df.sample(frac=percentage, random_state=42)

def get_datasets(data_dir, percentage=0.1):
    """Loads and preprocesses the datasets."""
    train_java_file = f'{data_dir}/train.java-cs.txt.java'
    train_cs_file = f'{data_dir}/train.java-cs.txt.cs'
    valid_java_file = f'{data_dir}/valid.java-cs.txt.java'
    valid_cs_file = f'{data_dir}/valid.java-cs.txt.cs'
    test_java_file = f'{data_dir}/test.java-cs.txt.java'
    test_cs_file = f'{data_dir}/test.java-cs.txt.cs'

    train_df = load_code_pairs(train_java_file, train_cs_file)

    if os.path.exists(valid_java_file) and os.path.exists(valid_cs_file):
        valid_df = load_code_pairs(valid_java_file, valid_cs_file)
    else:
        print("Validation files not found. Using training data for validation.")
        valid_df = train_df

    if os.path.exists(test_java_file) and os.path.exists(test_cs_file):
        test_df = load_code_pairs(test_java_file, test_cs_file)
    else:
        print("Test files not found. Using training data for testing.")
        test_df = train_df

    train_df_reduced = reduce_dataset_size(train_df, percentage)
    valid_df_reduced = reduce_dataset_size(valid_df, percentage)
    test_df_reduced = reduce_dataset_size(test_df, percentage)

    train_dataset = Dataset.from_pandas(train_df_reduced)
    valid_dataset = Dataset.from_pandas(valid_df_reduced)
    test_dataset = Dataset.from_pandas(test_df_reduced)

    return train_dataset, valid_dataset, test_dataset