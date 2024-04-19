import json
import ast
import sys
from pathlib import Path
from datetime import datetime
import os
import uuid

def abs_path(relative_path):
    base_path = Path(__file__).parent
    return (base_path / relative_path).resolve()

def organise_premises_into_pairs(arg_scheme_label, segmented_premises):
    paired_up_premises = {"train": [], "dev": [], "test": []}

    # Process each of train/dev/test
    for k,premises in segmented_premises.items():
        arg_scheme_premises = [v for v in premises if v["argumentation scheme label"] == arg_scheme_label]

        pairs_of_premises = []
        index_of_premise = 0
        for outer_premise in arg_scheme_premises:
            outer_premise_value, outer_argument_id = outer_premise["premise_value"], outer_premise["argument_id"]

            index_of_inner_premise = 0
            for inner_premise in arg_scheme_premises:
                inner_premise_value, inner_argument_id = inner_premise["premise_value"], inner_premise["argument_id"]

                if index_of_inner_premise > index_of_premise:

                    if ((outer_premise["is nlas extra"] or inner_premise["is nlas extra"]) and outer_argument_id != inner_argument_id):
                        continue

                    pairs_of_premises.append({
                        "text 1": outer_premise_value,
                        "text 2": inner_premise_value,
                        "are texts from same argument": int(outer_argument_id == inner_argument_id),
                        "argument ids": f'{outer_argument_id} {inner_argument_id}',
                        "topics": f'{outer_premise["topic"]} - {inner_premise["topic"]}',
                        "argumentation scheme label": arg_scheme_label,
                        "argumentation scheme": outer_premise["argumentation scheme"],
                        "is nlas extra": f'{outer_premise["is nlas extra"]} - {inner_premise["is nlas extra"]}'
                    })
                index_of_inner_premise += 1
            
            index_of_premise += 1
        
        paired_up_premises[k] = pairs_of_premises
    
    return paired_up_premises

def add_nlas_extra_positive_pairs(pairs_of_premises, arg_scheme_label, nlas_extra_scheme_premises):
    index_of_premise = 0
    for outer_premise in nlas_extra_scheme_premises:
        outer_premise_value, outer_argument_id = outer_premise["premise_value"], outer_premise["argument_id"]

        distance_from_outer_topic = 0
        index_of_inner_premise = 0
        for inner_premise in nlas_extra_scheme_premises:
            inner_premise_value, inner_argument_id = inner_premise["premise_value"], inner_premise["argument_id"]

            if index_of_inner_premise > index_of_premise and outer_argument_id == inner_argument_id:
    
                pairs_of_premises.append({
                    "text 1": outer_premise_value,
                    "text 2": inner_premise_value,
                    "are texts from same argument": int(outer_argument_id == inner_argument_id),
                    "argument ids": f'{outer_argument_id} {inner_argument_id}',
                    "topics": f'{outer_premise["topic"]} - {inner_premise["topic"]}',
                    "argumentation scheme label": arg_scheme_label,
                    "argumentation scheme": outer_premise["argumentation scheme"]
                })
            index_of_inner_premise += 1
        
        index_of_premise += 1

    pairs_of_premises.sort(key=lambda x: x["are texts from same argument"])

    # print(f'Processed from extra {index_of_premise} premises/conclusions for scheme {arg_scheme_label}')

    return pairs_of_premises

# def segment_data(pairs_of_premises):
    
#     negative_examples = [v for v in pairs_of_premises if v["are texts from same argument"] == 0]
#     positive_examples = [v for v in pairs_of_premises if v["are texts from same argument"] == 1]

#     final_data = {"train": [], "dev": [], "test": []}

#     train_positive = positive_examples[0 : int(0.8*len(positive_examples))]
#     dev_positive = positive_examples[len(train_positive) : len(train_positive) + int(0.1*len(positive_examples))]
#     test_positive = positive_examples[len(dev_positive + train_positive) : ]
#     final_data["train"].extend(train_positive)
#     final_data["dev"].extend(dev_positive)
#     final_data["test"].extend(test_positive)

#     train_negative = negative_examples[0 : int(0.8*len(negative_examples))]
#     dev_negative = negative_examples[len(train_negative) : len(train_negative) + int(0.1*len(negative_examples))]
#     test_negative = negative_examples[len(dev_negative + train_negative) : ]
#     final_data["train"].extend(train_negative)
#     final_data["dev"].extend(dev_negative)
#     final_data["test"].extend(test_negative)

#     return final_data

# print(f'From {num_of_arguments_processed} arguments, extracted {sum(len(v) for _,v in premises_by_arg_scheme.items())} premises (train: {len(final_data["train"])}, dev: {len(final_data["dev"])}, test: {len(final_data["test"])}), discarded {len(discarded_items)} arguments')

def calculate_segmented_data_metrics(data):
    metrics = {"train":{}, "dev":{}, "test":{}}
    metrics["train"]["correctly paired premises"] = len([v for v in data["train"] if v["are texts from same argument"] == 1])
    metrics["train"]["incorrectly paired premises"] = len([v for v in data["train"] if v["are texts from same argument"] == 0])

    metrics["dev"]["correctly paired premises"] = len([v for v in data["dev"] if v["are texts from same argument"] == 1])
    metrics["dev"]["incorrectly paired premises"] = len([v for v in data["dev"] if v["are texts from same argument"] == 0])
    
    metrics["test"]["correctly paired premises"] = len([v for v in data["test"] if v["are texts from same argument"] == 1])
    metrics["test"]["incorrectly paired premises"] = len([v for v in data["test"] if v["are texts from same argument"] == 0])

    return metrics

def run_module(dataset_path, arg_scheme_label, run_id):
    with open(dataset_path, "r", encoding="utf-8") as read_file:
        segmented_premises = json.load(read_file)
    # with open(dataset_path, "r", encoding="utf-8") as read_file:
    #     nlas_extra_premises = json.load(read_file)
    # with open(labels_map_path, "r") as file:
    #     labels_map = json.load(file)

    pairs_of_premises = organise_premises_into_pairs(arg_scheme_label, segmented_premises)
    # print(len(pairs_of_premises))
    # pairs_of_premises = add_nlas_extra_positive_pairs(pairs_of_premises, arg_scheme_label, nlas_extra_scheme_premises)
    # print(len(pairs_of_premises))

    segmented_metrics = calculate_segmented_data_metrics(pairs_of_premises)
    
    outputs_path = abs_path(f'outputs/{run_id}')
    os.makedirs(f'{outputs_path}/data', exist_ok=True)
    with open(abs_path(f'{outputs_path}/data/stage-2-scheme-{arg_scheme_label}-segmented-data.json'), "w") as file:
        json.dump(pairs_of_premises, file, indent=2)
    os.makedirs(f'{outputs_path}/metrics', exist_ok=True)
    with open(abs_path(f'{outputs_path}/metrics/stage-2-scheme-{arg_scheme_label}-segmented-data-metrics.json'), "w") as file:
        json.dump(segmented_metrics, file, indent=2)

if __name__ == "__main__":
    arg_scheme_label = int(sys.argv[1])
    dataset_path = sys.argv[2]
    existing_run_id = sys.argv[3]

    run_module(dataset_path, arg_scheme_label, existing_run_id)
    