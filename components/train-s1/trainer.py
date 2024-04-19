import json
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, AutoConfig, DataCollatorWithPadding
from huggingface_hub import HfFolder, notebook_login
import numpy as np
import evaluate
from pathlib import Path

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

def tokenize(examples):
    text = examples["text"]
    # print(examples)
    
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
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

if __name__ == "__main__":

    dataset_dir = abs_path(f'transformed-datasets/stage-1-final_data.json')
    output_dir = abs_path(f'models/')
    base_model_name = 'distilbert-base-uncased'
    num_of_epochs = 10
    num_of_labels = 20

    dataset = load_dataset(dataset_dir)
    base_model = load_base_model(base_model_name, num_of_labels)
    tokenizer = load_tokenizer(base_model_name, base_model)
    data_collator = load_data_collator(tokenizer)
    
    tokenized_dataset = dataset.map(tokenize, batched=True)
    
    trainer = train_model(base_model, tokenizer, tokenized_dataset, data_collator, output_dir, num_of_epochs)

    trainer.save_model(output_dir)
