import json
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, AutoConfig, DataCollatorWithPadding
from huggingface_hub import HfFolder, notebook_login
import numpy as np
import evaluate
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
from transformers import TextClassificationPipeline
import os
from pathlib import Path
import sys

def abs_path(relative_path):
    base_path = Path(__file__).parent
    print(relative_path)
    return (base_path / relative_path).resolve()

def load_dataset(file):
    with open(file, "r") as filehandle:
        print(file)
        json_data = json.load(filehandle)

    flattened_data = {
            'text1': [item["text 1"] for item in json_data],
            'text2': [item["text 2"] for item in json_data],
            'label': [item["are texts from same argument"] for item in json_data]
    }

    return DatasetDict({
        'dev': Dataset.from_dict(flattened_data)
    })

def load_base_model(base_model_name, label_count):
    return AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=label_count, ignore_mismatched_sizes=True)

tokenizer_2 = None
def load_tokenizer(base_model_name, base_model):
    tokenizer_2 = AutoTokenizer.from_pretrained(base_model_name)
    return tokenizer_2

def tokenization(examples):
    tokenized_inputs = tokenizer_2(
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

def load_local_model(path):
    return AutoModelForSequenceClassification.from_pretrained(path)

def run_module(stage_2_model_path, dataset_path, scheme_label, run_id):
    base_model_name = 'distilbert-base-uncased'
    trained_model_2 = load_local_model(stage_2_model_path)
    global tokenizer_2
    tokenizer_2 = load_tokenizer(base_model_name, trained_model_2)

    #TODO: text1, text2? Might need to generate the predictions for manual calculations of scores
    # pipe = TextClassificationPipeline(model=trained_model_2, tokenizer=tokenizer_2, top_k=1)
    # dataset_with_predictions_2 = []
    # for item in stage_2_dataset:
    #     premise_pair = f'{item["text 1"]}[SEP]{item["text 2"]}'
    #     prediction = pipe(premise_pair)[0][0]
    #     item["predicted are-from-same-argument label"] = int(prediction['label'].lstrip('LABEL_'))
    #     dataset_with_predictions_2.append(item)

    # with open(f'{outputs_path}/data/scheme-0-predictions.json', "w") as file:
    #     json.dump(dataset_with_predictions_2, file, indent=2)

    dataset = load_dataset(dataset_path)

    tokenized_dataset = dataset.map(tokenization, batched=True)

    data_collator = load_data_collator(tokenizer_2)

    trainer = Trainer(trained_model_2, tokenizer=tokenizer_2, data_collator=data_collator)

    dev_predictions_data = trainer.predict(tokenized_dataset['dev'])
    dev_predictions = np.argmax(dev_predictions_data.predictions, axis=-1)
    dev_macro_f1_with_others = precision_recall_fscore_support(tokenized_dataset['dev']['label'], dev_predictions, average='macro')
    dev_macro_f1 = f1_score(tokenized_dataset['dev']['label'], dev_predictions, average='macro')
    dev_binary_f1 = f1_score(tokenized_dataset['dev']['label'], dev_predictions)
    dev_confusion_matrix = confusion_matrix(tokenized_dataset['dev']['label'], dev_predictions)

    metrics = {"dev": {}}
    metrics["dev"]["Macro F1 with other metrics"] = dev_macro_f1_with_others
    metrics["dev"]["Macro F1"] = dev_macro_f1
    metrics["dev"]["Binary F1"] = dev_binary_f1
    metrics["dev"]["Confusion matrix"] = np.array2string(dev_confusion_matrix).split("\n")
    outputs_path = abs_path(f'outputs/{run_id}')
    os.makedirs(f'{outputs_path}/metrics', exist_ok=True)
    with open(f'{outputs_path}/metrics/stage-2-scheme-{scheme_label}-dev-metrics.json', "w") as file:
        json.dump(metrics, file, indent=2)

if __name__ == "__main__":
    stage_2_model_path = sys.argv[1]
    dataset_path = sys.argv[2]
    scheme_label = sys.argv[3] 
    run_id = sys.argv[4]
    print(stage_2_model_path)
    
    run_module(stage_2_model_path, dataset_path, scheme_label, run_id)

# python inference-pipeline/s2-inference/s2.py \
# modules/train-s2/outputs/run_5-segment-by-argument/models/stage-2-scheme-1 \
# inference-pipeline/reunite-conclusions/outputs/pipeline_1/data/scheme-1-pairs.json \
# 1 \
# pipeline_1

# python inference-pipeline/s2-inference/s2.py \
# modules/train-s2/outputs/run_5-segment-by-argument/models/stage-2-scheme-1 \
# inference-pipeline/pair-up-and-separate/outputs/pipeline_2_no_conclusions/data/scheme-1-pairs.json \
# 1 \
# pipeline_2_no_conclusions