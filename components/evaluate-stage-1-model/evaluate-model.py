import json
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, AutoConfig, DataCollatorWithPadding
from huggingface_hub import HfFolder, notebook_login
import numpy as np
import evaluate
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import os
import sys
import uuid
from datetime import datetime
from contextlib import redirect_stdout
from transformers import TextClassificationPipeline

def abs_path(relative_path):
    base_path = Path(__file__).parent
    return (base_path / relative_path).resolve()

def load_dataset(file):
    with open(file) as filehandle:
        json_data = json.load(filehandle)

    flattened_data = {
        'train': {
            'text': [item["premise value"] for item in json_data['train']],
            'label': [item["argumentation scheme label"] for item in json_data['train']]
        },
        'dev': {
            'text': [item["premise value"] for item in json_data['dev']],
            'label': [item["argumentation scheme label"] for item in json_data['dev']]
        },
        'test': {
            'text': [item["premise value"] for item in json_data['test']],
            'label': [item["argumentation scheme label"] for item in json_data['test']]
        }
    }

    formatted_data = DatasetDict({
        'train': Dataset.from_dict(flattened_data['train']),
        'dev': Dataset.from_dict(flattened_data['dev']),
        'test': Dataset.from_dict(flattened_data['test'])
    })

    return formatted_data

def load_base_model(base_model_name, label_count):
    return AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=label_count, ignore_mismatched_sizes=True)

def load_tokenizer(base_model_name, base_model):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # print(tokenizer.pad_token)

    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     base_model_name.resize_token_embeddings(len(tokenizer))

    return tokenizer

def tokenization(examples):
    text = examples["text"]
    # print(examples)
    
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        padding=True,
        max_length=512
    )

    return tokenized_inputs

def load_data_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer)

def load_local_model(path):
    return AutoModelForSequenceClassification.from_pretrained(path)

tokenizer = None

def run_module(dataset_path, local_model_path, existing_run_id):

    if existing_run_id == None:
        existing_run_id = f'{(datetime.now()).strftime("%y%m%d%H%M%S")}_{uuid.uuid4()}'
    outputs_path = abs_path(f'outputs/{existing_run_id}')
    os.makedirs(outputs_path, exist_ok=True)

    os.makedirs(f'{outputs_path}/logs', exist_ok=True)
    with open(f'{outputs_path}/logs/stage-1-dev-eval-log', 'w') as f:
        with redirect_stdout(f):
            print("logs placeholder")

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
    dev_macro_f1 = precision_recall_fscore_support(tokenized_dataset['dev']['label'], dev_predictions, average='macro')
    dev_confusion_matrix = confusion_matrix(tokenized_dataset['dev']['label'], dev_predictions)
    
    os.makedirs(f'{outputs_path}/metrics', exist_ok=True)
    with open(f'{outputs_path}/metrics/stage-1-dev-metrics', "w") as file:
        file.write(f"Macro F1: {dev_macro_f1} \n")
        file.write("Confusion matrix:\n")
        file.write(np.array2string(dev_confusion_matrix, separator=', '))
    
    df_cfm = pd.DataFrame(dev_confusion_matrix)
    plt.figure(figsize = (10,7))
    cfm_plot = sn.heatmap(df_cfm, annot=True, fmt='d')
    os.makedirs(f'{outputs_path}/metrics', exist_ok=True)
    cfm_plot.figure.savefig(f"{outputs_path}/metrics/stage-1-dev-cfm.png")

    # pipe = TextClassificationPipeline(model=trained_model, tokenizer=tokenizer, top_k=1)
    # with open(dataset_path, 'r') as input_file:
    #     input_dataset = json.load(input_file)
    #     dataset_with_predictions = {'dev': []}
    #     for item in input_dataset['dev']:
    #         prediction = pipe(item["premise value"])[0][0]
    #         item["predicted argumentation scheme label"] = int(prediction['label'].lstrip('LABEL_'))
    #         item["predicted argumentation scheme score"] = round(prediction['score'], 2)
    #         dataset_with_predictions['dev'].append(item)
    #         print(item)
    # os.makedirs(f'{outputs_path}/metrics', exist_ok=True)
    # with open(f'{outputs_path}/metrics/stage-1-dev-predictions', "w") as file:
    #     json.dump(dataset_with_predictions, file, indent=2)

if __name__ == "__main__":
    dataset_path = sys.argv[1]
    local_model_path = sys.argv[2]
    existing_run_id = sys.argv[3] if len(sys.argv) > 3 else None

    run_module(dataset_path, local_model_path, existing_run_id)
