import torch
from datasets import load_dataset
import pandas as pd

import model_config
import data_processing
import finetune

model_name = 'meta-llama/Llama-2-7b-hf'
load_in_4bit = True
bnb_4bit_use_double_quant = True
bnb_4bit_quant_type = 'nf4'
bnb_4bit_compute_dtype = torch.bfloat16

bnb_config = model_config.create_bnb_config(load_in_4bit, bnb_4bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype)

model, tokenizer = model_config.load_model(model_name, bnb_config)

df1 = pd.read_csv('/content/drive/MyDrive/SBIC.v2.agg.trn.csv')
df1['Instruction'] = 'Categorize the given post into one of the 2 categories:\n\nHaving Biased Implications\nNo biased Implications'
df1 = df1[['Instruction', 'post', 'hasBiasedImplication']]
df1 = df1.rename(columns={'hasBiasedImplication': 'category', 'post': 'text'})
df1['text'] = df1['text'].apply(data_processing.clean_string)
df1.to_csv('/content/drive/MyDrive/cleaned_implications.csv', index=False)
dataset_name = "/content/drive/MyDrive/cleaned_implications.csv"
dataset = load_dataset("csv", data_files=dataset_name, split="train")

seed = 33

max_length = model_config.get_max_length(model)
preprocessed_dataset = data_processing.preprocess_dataset(tokenizer, max_length, seed, dataset)

lora_r = 16
lora_alpha = 64
lora_dropout = 0.1
bias = "none"
task_type = "CAUSAL_LM"

output_dir = "./results"
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
learning_rate = 2e-4
optim = "paged_adamw_32bit"
max_steps = 20
warmup_steps = 2
fp16 = True
logging_steps = 1

finetune.fine_tune(model, tokenizer, preprocessed_dataset, 
                    lora_r, lora_alpha, lora_dropout, bias, 
                    task_type, per_device_train_batch_size, 
                    gradient_accumulation_steps, warmup_steps, 
                    max_steps, learning_rate, fp16, logging_steps, 
                    output_dir, optim)
