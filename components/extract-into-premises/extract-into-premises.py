import json
import ast
import sys
from pathlib import Path
from datetime import datetime
import uuid
import os

def abs_path(relative_path):
    base_path = Path(__file__).parent
    return (base_path / relative_path).resolve()

def extract_premises(segmented_corpus):
    all_premises = {"train":[], "dev":[], "test":[]}
    all_premises["train"].extend(extract_premises_in_segmentation(segmented_corpus["train"]))
    all_premises["dev"].extend(extract_premises_in_segmentation(segmented_corpus["dev"]))
    all_premises["test"].extend(extract_premises_in_segmentation(segmented_corpus["test"]))

    return all_premises
    
def extract_premises_in_segmentation(corpus):

    premises_by_arg_scheme = []
    discarded_items = []
    num_of_arguments_processed = 0

    for corpusItem in corpus:
        argument = corpusItem["argument"]

        if (type(argument) is str):
            try:
                argument = ast.literal_eval(argument)
            # Some arguments have invalid JSON (text appended after the closing bracket or an apostrophe inside single quotes)
            except SyntaxError:
                # print(f'Discarding #{key}')
                discarded_items.append(corpusItem)
                continue
        
        # # Some arguments only have a single premise and a conclusion
        # if len(argument.items()) < 3:
        #     print(f'Discarding {key}')
        #     discarded_items[key] = corpusItem
        #     continue

        # NOTE: Conclusions are included in stage 2 training        
        for premise_key, premise_value in argument.items():
            premise_with_arg_id = {
                "premise_value": premise_value,
                "argumentation scheme": corpusItem["argumentation scheme"],
                "argumentation scheme label": corpusItem["argumentation scheme label"],
                "argument_id": corpusItem["argument id"],
                "topic": corpusItem["topic"],
                "is conclusion": premise_key.lower() == "conclusion",
                "is nlas extra": corpusItem["is nlas extra"]
            }
            premises_by_arg_scheme.append(premise_with_arg_id)

        num_of_arguments_processed += 1

    return premises_by_arg_scheme

# print(f'From {num_of_arguments_processed} arguments, extracted {sum(len(v) for _,v in premises_by_arg_scheme.items())} premises (train: {len(final_data["train"])}, dev: {len(final_data["dev"])}, test: {len(final_data["test"])}), discarded {len(discarded_items)} arguments')

def calculate_premise_metrics(segmented_premises):
    all_topics = {"train":{}, "dev":{}, "test":{}}
    all_schemes = {"train":{}, "dev":{}, "test":{}}

    for k,_ in all_topics.items():
        topics = {}
        for premise in segmented_premises[k]:
            if premise["topic"] in topics:
                topics[premise["topic"]] += 1
            else:
                topics[premise["topic"]] = 0
        all_topics[k] = topics

    all_schemes = {"train":{}, "dev":{}, "test":{}}
    for k,_ in all_schemes.items():
        arg_schemes = {}
        for premise in segmented_premises[k]:
            if premise["argumentation scheme label"] in arg_schemes:
                arg_schemes[premise["argumentation scheme label"]] += 1
            else:
                arg_schemes[premise["argumentation scheme label"]] = 0
        all_schemes[k] = arg_schemes

    return all_topics, all_schemes

def run_module(dataset_path, existing_run_id):
    with open(dataset_path, "r", encoding="utf-8") as read_file:
        segmented_arguments = json.load(read_file)

    all_premises = extract_premises(segmented_arguments)
    topics, arg_schemes = calculate_premise_metrics(all_premises)

    outputs_path = abs_path(f'outputs/{existing_run_id}')
    os.makedirs(f'{outputs_path}/data', exist_ok=True)
    # with open(f"{outputs_path}/data/stage-2-discarded-items.json", "w") as file:
    #     json.dump(discarded_items, file, indent=2)
    with open(f"{outputs_path}/data/premises.json", "w") as file:
        json.dump(all_premises, file, indent=2)

    os.makedirs(f'{outputs_path}/metrics', exist_ok=True)
    with open(f"{outputs_path}/metrics/topics.json", "w") as file:   
        json.dump(topics, file, indent=2)
    with open(f"{outputs_path}/metrics/arg_schemes.json", "w") as file:   
        json.dump(arg_schemes, file, indent=2)

if __name__ == "__main__":
    dataset_path = sys.argv[1]
    existing_uuid = sys.argv[2]

    run_module(dataset_path, existing_uuid)
    
    # DATASET_PATH = abs_path("../datasets/nlas_eng.json")
    # LABELS_MAP_PATH = abs_path("../datasets/labels_map.json")
