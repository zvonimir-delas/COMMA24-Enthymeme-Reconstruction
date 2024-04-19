import json
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, AutoConfig, DataCollatorWithPadding
from huggingface_hub import HfFolder, notebook_login
import numpy as np
import evaluate
import sys
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from datetime import datetime
import os
import uuid
from contextlib import redirect_stdout

def abs_path(relative_path):
    base_path = Path(__file__).parent
    return (base_path / relative_path).resolve()

def load_dataset(file):
    with open(file) as filehandle:
        json_data = json.load(filehandle)

    flattened_data = {
        'train': {
            'text1': [item["text 1"] for item in json_data['train']],
            'text2': [item["text 2"] for item in json_data['train']],
            'label': [item["are texts from same argument"] for item in json_data['train']]
        },
        'dev': {
            'text1': [item["text 1"] for item in json_data['dev']],
            'text2': [item["text 2"] for item in json_data['dev']],
            'label': [item["are texts from same argument"] for item in json_data['dev']]
        },
        'test': {
            'text1': [item["text 1"] for item in json_data['test']],
            'text2': [item["text 2"] for item in json_data['test']],
            'label': [item["are texts from same argument"] for item in json_data['test']]
        }
    }

    formatted_data = DatasetDict({
        'train': Dataset.from_dict(flattened_data['train']),
        'dev': Dataset.from_dict(flattened_data['dev']),
        'test': Dataset.from_dict(flattened_data['test'])
    })

    # print(formatted_data)

    return formatted_data

def load_base_model(base_model_name, label_count):
    return AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=label_count, ignore_mismatched_sizes=True)

def load_tokenizer(base_model_name, base_model):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.truncation_side = "left"

    # print(tokenizer.pad_token)

    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     base_model_name.resize_token_embeddings(len(tokenizer))

    return tokenizer

def tokenization(examples):
    tokenized_inputs = tokenizer(
        examples["text1"],
        examples["text2"],
        return_tensors="np",
        truncation=True,
        padding=True,
        max_length=512
    )

    return tokenized_inputs

def load_data_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    accuracy = evaluate.load("f1")

    return accuracy.compute(predictions=predictions, references=labels, average='macro')

def train_model(base_model, tokenizer, dataset, data_collator, output_dir, num_of_epochs):

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=1e-5,
        weight_decay=0.01,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        num_train_epochs=num_of_epochs,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        fp16=True
    )

    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['dev'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    return trainer

# Todo: must be a global variable
tokenizer = None

def run_module(arg_scheme_label, dataset_path, existing_run_id=None):

    if existing_run_id == None:
        existing_run_id = f'{(datetime.now()).strftime("%y%m%d%H%M%S")}_{uuid.uuid4()}'
    outputs_path = abs_path(f'outputs/{existing_run_id}')
    os.makedirs(outputs_path, exist_ok=True)

    os.makedirs(f'{outputs_path}/models', exist_ok=True)
    model_output_dir = f'{outputs_path}/models/stage-2-scheme-{arg_scheme_label}'
    base_model_name = 'distilbert-base-uncased'
    label_count = 2
    num_of_epochs = 8

    dataset = load_dataset(dataset_path).shuffle(seed=42)
    base_model = load_base_model(base_model_name, label_count)
    global tokenizer
    tokenizer = load_tokenizer(base_model_name, base_model)
    data_collator = load_data_collator(tokenizer)
    
    tokenized_dataset = dataset.map(tokenization, batched=True)
    
    os.makedirs(f'{outputs_path}/logs', exist_ok=True)
    with open(f'{outputs_path}/logs/stage-2-scheme-{arg_scheme_label}-training-log', 'w') as f:
        with redirect_stdout(f):
            trainer = train_model(base_model, tokenizer, tokenized_dataset, data_collator, model_output_dir, num_of_epochs)

    trainer.save_model(model_output_dir)

if __name__ == "__main__":
    arg_scheme_label = int(sys.argv[1])
    dataset_path = sys.argv[2]
    existing_run_id = sys.argv[3] if len(sys.argv) > 3 else None

    run_module(arg_scheme_label, dataset_path, existing_run_id)
