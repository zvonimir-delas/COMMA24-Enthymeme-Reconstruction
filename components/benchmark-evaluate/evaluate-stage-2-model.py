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

def load_local_model(path):
    return AutoModelForSequenceClassification.from_pretrained(path)

# Todo: must be a global variable
tokenizer = None

def load_data_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer)

def run_module(dataset_path, local_model_path, existing_run_id):

    outputs_path = abs_path(f'outputs/{existing_run_id}')
    os.makedirs(outputs_path, exist_ok=True)

    dataset = load_dataset(dataset_path)
    base_model_name = 'distilbert-base-uncased'
    trained_model = load_local_model(local_model_path)
    global tokenizer
    tokenizer = load_tokenizer(base_model_name, trained_model)
    data_collator = load_data_collator(tokenizer)

    tokenized_dataset = dataset.map(tokenization, batched=True)
    trainer = Trainer(trained_model, tokenizer=tokenizer, data_collator=data_collator)

    dev_predictions_data = trainer.predict(tokenized_dataset['dev'])
    dev_predictions = np.argmax(dev_predictions_data.predictions, axis=-1)
    dev_macro_f1_with_others = precision_recall_fscore_support(tokenized_dataset['dev']['label'], dev_predictions, average='macro')
    dev_macro_f1 = f1_score(tokenized_dataset['dev']['label'], dev_predictions, average='macro')
    dev_binary_f1 = f1_score(tokenized_dataset['dev']['label'], dev_predictions)
    dev_confusion_matrix = confusion_matrix(tokenized_dataset['dev']['label'], dev_predictions)
    
    # print(f'Stage 1 - Macro F1: {dev_macro_f1}')
    # print('Stage 1 - Confusion matrix:')
    # print(dev_confusion_matrix)
    
    os.makedirs(f'{outputs_path}/metrics', exist_ok=True)
    with open(f'{outputs_path}/metrics/benchmark-dev-metrics', "w") as file:
        file.write(f"Macro F1 with other metrics: {dev_macro_f1_with_others} \n")
        file.write(f"Macro F1: {dev_macro_f1} \n")
        file.write(f"Binary F1: {dev_binary_f1} \n")
        file.write("Confusion matrix:\n")
        file.write(np.array2string(dev_confusion_matrix, separator=', '))
    
    # df_cfm = pd.DataFrame(dev_confusion_matrix)
    # plt.figure(figsize = (10,7))
    # cfm_plot = sn.heatmap(df_cfm, annot=True, fmt='d')
    # os.makedirs(f'{outputs_path}/metrics', exist_ok=True)
    # cfm_plot.figure.savefig(f"{outputs_path}/metrics/stage-2-scheme-{arg_scheme_label}-dev-cfm.png")

if __name__ == "__main__":
    dataset_path = sys.argv[1]
    model_path = sys.argv[2]
    existing_run_id = sys.argv[3]

    run_module(dataset_path, model_path, existing_run_id)

# python modules/benchmark-evaluate/evaluate-stage-2-model.py modules/benchmark-combine-pairs/outputs/benchmark_1/data/combined-pairs-from-all-schemes.json modules/benchmark-train/outputs/benchmark_1/models benchmark_1
# python modules/benchmark-evaluate/evaluate-stage-2-model.py modules/benchmark-combine-for-inference/outputs/benchmark_1/data/combined-pairs-from-all-schemes-with-all-conclusions.json modules/benchmark-train/outputs/benchmark_1/models benchmark_1_with_all_conclusions
# python modules/benchmark-evaluate/evaluate-stage-2-model.py modules/benchmark-combine-for-inference/outputs/benchmark_2_no_conclusions/data/combined-pairs-from-all-schemes-with-all-conclusions.json modules/benchmark-train/outputs/benchmark_1/models benchmark_2_no_conclusions
# python modules/benchmark-evaluate/evaluate-stage-2-model.py modules/benchmark-pair-up/outputs/pipeline_3_no_conclusions/data/pairs-from-all-schemes.json modules/benchmark-train/outputs/benchmark_1/models benchmark_3_no_conclusions
