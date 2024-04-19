import json, sys, os
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, AutoConfig, DataCollatorWithPadding
from huggingface_hub import HfFolder, notebook_login
import numpy as np
import evaluate
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
from transformers import TextClassificationPipeline
from pathlib import Path

def abs_path(relative_path):
    base_path = Path(__file__).parent
    return (base_path / relative_path).resolve()

tokenizer = None
def load_tokenizer(base_model_name, base_model):
    tokenizer_2 = AutoTokenizer.from_pretrained(base_model_name)
    return tokenizer_2

def tokenization(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        return_tensors="np",
        truncation=True,
        padding=True,
        max_length=512
    )

    return tokenized_inputs

def load_local_model(path):
    return AutoModelForSequenceClassification.from_pretrained(path)

def run_module(dev_premises_without_conclusions_path, stage_1_model_path, run_id):
    base_model_name = 'distilbert-base-uncased'
    trained_model = load_local_model(stage_1_model_path)
    tokenizer = load_tokenizer(base_model_name, trained_model)

    with open(dev_premises_without_conclusions_path, "r") as filehandle:
        dev_premises_without_conclusions = json.load(filehandle)

    pipe = TextClassificationPipeline(model=trained_model, tokenizer=tokenizer, top_k=1)
    dataset_with_predictions = []
    for item in dev_premises_without_conclusions:
        prediction = pipe(item["premise_value"])[0][0]
        item["predicted argumentation scheme label"] = int(prediction['label'].lstrip('LABEL_'))
        item["predicted argumentation scheme score"] = round(prediction['score'], 2)
        dataset_with_predictions.append(item)
    
    outputs_path = abs_path(f'outputs/{run_id}')
    os.makedirs(f'{outputs_path}/data', exist_ok=True)
    with open(f'{outputs_path}/data/stage-1-dev-predictions.json', "w") as file:
        json.dump(dataset_with_predictions, file, indent=2)

    # Note: this should be the same as S1 evaluation metrics
    flattened_dev_with_predictions = {
        'predictions': [item["argumentation scheme label"] for item in dataset_with_predictions],
        'baseline': [item["predicted argumentation scheme label"] for item in dataset_with_predictions]
    }

    #TODO: not sure what dev predictions does
    # dev_predictions = np.argmax(flattened_dev_with_predictions['predictions'], axis=-1)
    dev_macro_f1_with_others = precision_recall_fscore_support(flattened_dev_with_predictions['baseline'], flattened_dev_with_predictions['predictions'], average='macro')
    dev_macro_f1 = f1_score(flattened_dev_with_predictions['baseline'], flattened_dev_with_predictions['predictions'], average='macro')
    # dev_binary_f1 = f1_score(flattened_dev_with_predictions['baseline'], flattened_dev_with_predictions['predictions'])
    dev_confusion_matrix = confusion_matrix(flattened_dev_with_predictions['baseline'], flattened_dev_with_predictions['predictions'])
    
    metrics = {"dev": {}}
    metrics["dev"]["Macro F1 with other metrics"] = dev_macro_f1_with_others
    metrics["dev"]["Macro F1"] = dev_macro_f1
    # metrics["dev"]["Binary F1"] = dev_binary_f1
    metrics["dev"]["Confusion matrix"] = np.array2string(dev_confusion_matrix).split("\n")
    os.makedirs(f'{outputs_path}/metrics', exist_ok=True)
    with open(f'{outputs_path}/metrics/stage-1-dev-predictions-metrics.json', "w") as file:
        json.dump(metrics, file, indent=2)

if __name__ == "__main__":
    dev_premises_without_conclusions_path = sys.argv[1]
    stage_1_model_path = sys.argv[2]
    run_id = sys.argv[3]

    run_module(dev_premises_without_conclusions_path, stage_1_model_path, run_id)

# python inference-pipeline/s1-inference/s1.py \
# inference-pipeline/separate-conclusions/outputs/pipeline_1/data/dev-premises-without-conclusions.json \
# stage-1/models \
# pipeline_1