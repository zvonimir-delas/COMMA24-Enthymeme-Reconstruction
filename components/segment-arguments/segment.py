import json, sys, os
from pathlib import Path

def abs_path(relative_path):
    base_path = Path(__file__).parent
    return (base_path / relative_path).resolve()

def segment_data(combined_arguments):
    final_data = {"train": [], "dev": [], "test": []}
    nlas_arguments = [v for v in combined_arguments if not v["is nlas extra"]]
    nlas_extra_arguments = [v for v in combined_arguments if v["is nlas extra"]]
    
    for scheme in range(20):
        arguments_in_scheme_nlas = [v for v in nlas_arguments if v["argumentation scheme label"] == scheme]
        train = arguments_in_scheme_nlas[0 : int(0.8*len(arguments_in_scheme_nlas))]
        dev = arguments_in_scheme_nlas[len(train) : len(train) + int(0.1*len(arguments_in_scheme_nlas))]
        test = arguments_in_scheme_nlas[len(train + dev) : ]
        final_data["train"].extend(train)
        final_data["dev"].extend(dev)
        final_data["test"].extend(test) 

        arguments_in_scheme_nlas_extra = [v for v in nlas_extra_arguments if v["argumentation scheme label"] == scheme]
        train = arguments_in_scheme_nlas_extra[0 : int(0.8*len(arguments_in_scheme_nlas_extra))]
        dev = arguments_in_scheme_nlas_extra[len(train) : len(train) + int(0.1*len(arguments_in_scheme_nlas_extra))]
        test = arguments_in_scheme_nlas_extra[len(train + dev) : ]
        final_data["train"].extend(train)
        final_data["dev"].extend(dev)
        final_data["test"].extend(test) 

    return final_data

def calculate_segmented_data_metrics(data):
    metrics = {"train":{}, "dev":{}, "test":{}}
    metrics["train"]["total arguments counts"] = len(data["train"])
    metrics["train"]["arguments from NLAS extra"] = len([v for v in data["train"] if v["is nlas extra"] == 0])
    metrics["train"]["arguments by label"] = {v:{"nlas": 0, "nlas extra": 0} for v in range(20)}
    for argument in data["train"]:
        if not argument["is nlas extra"]:
            metrics["train"]["arguments by label"][argument["argumentation scheme label"]]["nlas"] += 1
        else:
            metrics["train"]["arguments by label"][argument["argumentation scheme label"]]["nlas extra"] += 1

    metrics["dev"]["total arguments counts"] = len(data["dev"])
    metrics["dev"]["arguments from NLAS extra"] = len([v for v in data["dev"] if v["is nlas extra"] == 0])
    metrics["dev"]["arguments by label"] = {v:{"nlas": 0, "nlas extra": 0} for v in range(20)}
    for argument in data["dev"]:
        if not argument["is nlas extra"]:
            metrics["dev"]["arguments by label"][argument["argumentation scheme label"]]["nlas"] += 1
        else:
            metrics["dev"]["arguments by label"][argument["argumentation scheme label"]]["nlas extra"] += 1

    metrics["test"]["total arguments counts"] = len(data["test"])
    metrics["test"]["arguments from NLAS extra"] = len([v for v in data["test"] if v["is nlas extra"] == 0])
    metrics["test"]["arguments by label"] = {v:{"nlas": 0, "nlas extra": 0} for v in range(20)}
    for argument in data["test"]:
        if not argument["is nlas extra"]:
            metrics["test"]["arguments by label"][argument["argumentation scheme label"]]["nlas"] += 1
        else:
            metrics["test"]["arguments by label"][argument["argumentation scheme label"]]["nlas extra"] += 1

    return metrics

def run_module(dataset_path, run_id):
    with open(dataset_path, "r", encoding="utf-8") as read_file:
        arguments = json.load(read_file)

    segmented_data = segment_data(arguments)

    segmented_metrics = calculate_segmented_data_metrics(segmented_data)
    
    outputs_path = abs_path(f'outputs/{run_id}')
    os.makedirs(f'{outputs_path}/data', exist_ok=True)
    with open(abs_path(f'{outputs_path}/data/segmented-arguments.json'), "w") as file:
        json.dump(segmented_data, file, indent=2)
    os.makedirs(f'{outputs_path}/metrics', exist_ok=True)
    with open(abs_path(f'{outputs_path}/metrics/segmented-arguments-metrics.json'), "w") as file:
        json.dump(segmented_metrics, file, indent=2)

if __name__ == "__main__":
    dataset_path = sys.argv[1]
    run_id = sys.argv[2]

    run_module(dataset_path, run_id)
