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

def run_module(stage_1_predictions_path, run_id):
    with open(stage_1_predictions_path, "r") as filehandle:
        dataset_with_predictions = json.load(filehandle)

    for scheme in range(20):
        premises_by_scheme = [premise for premise in dataset_with_predictions if premise["predicted argumentation scheme label"] == scheme]
        pairs_of_premises = []
        index_of_premise = 0
        predicted_scheme = scheme
        for outer_premise in premises_by_scheme:
            outer_premise_value, outer_argument_id = outer_premise["premise_value"], outer_premise["argument_id"]

            index_of_inner_premise = 0
            for inner_premise in premises_by_scheme:
                inner_premise_value, inner_argument_id = inner_premise["premise_value"], inner_premise["argument_id"]

                if index_of_inner_premise > index_of_premise:

                    # if ((outer_premise["is nlas extra"] or inner_premise["is nlas extra"]) and outer_argument_id != inner_argument_id):
                    #     continue

                    pairs_of_premises.append({
                        "text 1": outer_premise_value,
                        "text 2": inner_premise_value,
                        "are texts from same argument": int(outer_argument_id == inner_argument_id),
                        "argument ids": f'{outer_argument_id} {inner_argument_id}',
                        "topics": f'{outer_premise["topic"]} - {inner_premise["topic"]}',
                        "predicted argumentation scheme label": predicted_scheme,
                        # "argumentation scheme": outer_premise["argumentation scheme"],
                        "is nlas extra": f'{outer_premise["is nlas extra"]} - {inner_premise["is nlas extra"]}'
                    })
                index_of_inner_premise += 1
            
            index_of_premise += 1
        
        outputs_path = abs_path(f'outputs/{run_id}')
        os.makedirs(f'{outputs_path}/data', exist_ok=True)
        with open(f'{outputs_path}/data/scheme-{predicted_scheme}-pairs.json', "w") as file:
            json.dump(pairs_of_premises, file, indent=2)
        
        metrics = {"dev": {}}
        metrics["dev"]["positive premise pairs"] = len([v for v in pairs_of_premises if v["are texts from same argument"] == 1])
        metrics["dev"]["negative premise pairs"] = len([v for v in pairs_of_premises if v["are texts from same argument"] == 0])
        os.makedirs(f'{outputs_path}/metrics', exist_ok=True)
        with open(f'{outputs_path}/metrics/scheme-{predicted_scheme}-input-metrics.json', "w") as file:
            json.dump(metrics, file, indent=2)

if __name__ == "__main__":
    stage_1_predictions_path = sys.argv[1]
    run_id = sys.argv[2]

    run_module(stage_1_predictions_path, run_id)

# python inference-pipeline/pair-up-and-separate/pair-sep.py \
# inference-pipeline/s1-inference/outputs/pipeline_1/data/stage-1-dev-predictions.json \
# pipeline_2_no_conclusions
