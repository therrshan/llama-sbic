from functools import partial
from transformers import AutoTokenizer
import re
from prompt_formatting import create_prompt_formats

def clean_string(s):
    s = s.replace('\n', '')
    s = re.sub(r'[^\w\s]', '', s)
    return s


def preprocess_batch(batch, tokenizer, max_length):

    return tokenizer(
        batch["text"],
        max_length = max_length,
        truncation = True,
    )

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str):

    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)

    _preprocessing_function = partial(preprocess_batch, max_length = max_length, tokenizer = tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched = True,
        remove_columns = ["Instruction", "text", "category", "Text"],
    )

    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    dataset = dataset.shuffle(seed = seed)

    return dataset