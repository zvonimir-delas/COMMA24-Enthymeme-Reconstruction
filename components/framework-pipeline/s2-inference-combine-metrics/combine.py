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

def run_module(path_to_eval_metrics, existing_run_id):
    outputs_path = abs_path(f'outputs/{existing_run_id}')
    os.makedirs(f'{outputs_path}/metrics', exist_ok=True)

    combined_metrics = {"Pipeline S2": {}}
    binary_f1_sum = 0
    macro_f1_sum = 0
    nonzero_binary_f1_counter = 0 

    for x in range(20):
        with open(f'{path_to_eval_metrics}/stage-2-scheme-{x}-dev-metrics.json', 'r') as file1:
            scheme_eval_data = json.load(file1)
            combined_metrics["Pipeline S2"][f'Scheme {x}'] = scheme_eval_data
            
            binary_f1 = scheme_eval_data["dev"]["Binary F1"]
            if binary_f1 > 0.0:
                binary_f1_sum += scheme_eval_data["dev"]["Binary F1"]
                macro_f1_sum += scheme_eval_data["dev"]["Macro F1"]
                nonzero_binary_f1_counter += 1

    combined_metrics["Pipeline S2"]["Avg across schemes"] = {
        "Binary F1": binary_f1_sum / nonzero_binary_f1_counter,
        "Macro F1": macro_f1_sum / nonzero_binary_f1_counter,
        "Non-zero binary F1 counter": nonzero_binary_f1_counter
    }
            
    with open(abs_path(f'{outputs_path}/metrics/combined-metrics.json'), "w") as outputfile:
        json.dump(combined_metrics, outputfile, indent=2)

if __name__ == "__main__":
    path_to_eval_metrics = sys.argv[1]
    run_id = sys.argv[2]

    run_module(path_to_eval_metrics, run_id)

# python inference-pipeline/s2-inference-combine-metrics/combine.py inference-pipeline/s2-inference/outputs/pipeline_2_no_conclusions/metrics pipeline_2_no_conclusions
    