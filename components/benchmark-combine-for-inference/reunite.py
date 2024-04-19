import sys, json, os
from pathlib import Path

def abs_path(relative_path):
    base_path = Path(__file__).parent
    return (base_path / relative_path).resolve()

def calculate_segmented_data_metrics(data):
    metrics = {"train":{}, "dev":{}, "test":{}}
    # metrics["train"]["correctly paired premises"] = len([v for v in data["train"] if v["are texts from same argument"] == 1])
    # metrics["train"]["incorrectly paired premises"] = len([v for v in data["train"] if v["are texts from same argument"] == 0])

    metrics["dev"]["correctly paired premises"] = len([v for v in data["dev"] if v["are texts from same argument"] == 1])
    metrics["dev"]["incorrectly paired premises"] = len([v for v in data["dev"] if v["are texts from same argument"] == 0])
    
    # metrics["test"]["correctly paired premises"] = len([v for v in data["test"] if v["are texts from same argument"] == 1])
    # metrics["test"]["incorrectly paired premises"] = len([v for v in data["test"] if v["are texts from same argument"] == 0])

    return metrics

def run_module(pairs_folder_path, run_id):
    combined_pairs = {"train": [], "dev": [], "test": []}
    for x in range(20):
        with open(f'{pairs_folder_path}/scheme-{x}-pairs.json', "r", encoding="utf-8") as read_file:
            pairs = json.load(read_file)
            # combined_pairs["train"].extend(pairs["train"])
            combined_pairs["dev"].extend(pairs)
            # combined_pairs["test"].extend(pairs["test"])

    outputs_path = abs_path(f'outputs/{run_id}')
    os.makedirs(f'{outputs_path}/data', exist_ok=True)
    with open(f"{outputs_path}/data/combined-pairs-from-all-schemes-with-all-conclusions.json", "w") as file:
        json.dump(combined_pairs, file, indent=2)

    metrics = calculate_segmented_data_metrics(combined_pairs)
    os.makedirs(f'{outputs_path}/metrics', exist_ok=True)
    with open(f"{outputs_path}/metrics/combined-pairs-from-all-schemes-with-all-conclusions-metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

if __name__ == "__main__":
    pairs_folder_path = sys.argv[1]
    run_id = sys.argv[2]

    run_module(pairs_folder_path, run_id)

# python modules/benchmark-combine-for-inference/reunite.py inference-pipeline/reunite-conclusions/outputs/pipeline_1/data benchmark_1
# python modules/benchmark-combine-for-inference/reunite.py inference-pipeline/pair-up-and-separate/outputs/pipeline_2_no_conclusions/data benchmark_2_no_conclusions