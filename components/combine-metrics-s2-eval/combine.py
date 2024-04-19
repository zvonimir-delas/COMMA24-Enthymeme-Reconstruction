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

def run_module(path_to_eval_metrics, path_to_segmentation_metrics, existing_run_id):
    outputs_path = abs_path(f'outputs/{existing_run_id}')
    os.makedirs(f'{outputs_path}/metrics', exist_ok=True)

    with open(abs_path(f'{outputs_path}/metrics/combined-metrics'), "w") as outputfile:
        for x in range(20):
            outputfile.write(f'---- STAGE 2 SCHEME {x}:\n')

            with open(f'{path_to_segmentation_metrics}/stage-2-scheme-{x}-segmented-data-metrics.json', 'r') as file2:
                outputfile.write('-- Segmentation metrics:\n')
                file2data = json.load(file2)
                outputfile.write(f'Train: pos: {file2data["train"]["correctly paired premises"]} neg: {file2data["train"]["incorrectly paired premises"]}\n')
                outputfile.write(f'Dev: pos: {file2data["dev"]["correctly paired premises"]} neg: {file2data["dev"]["incorrectly paired premises"]}\n')
                outputfile.write(f'Test: pos: {file2data["test"]["correctly paired premises"]} neg: {file2data["test"]["incorrectly paired premises"]}\n')

            with open(f'{path_to_eval_metrics}/stage-2-scheme-{x}-dev-metrics', 'r') as file1:
                outputfile.write('-- Evaluation (DEV) metrics:\n')
                outputfile.write(file1.read())
            
            outputfile.write('\n\n-------------------------------\n\n')
if __name__ == "__main__":
    path_to_eval_metrics = sys.argv[1]
    path_to_segmentation_metrics = sys.argv[2]
    existing_run_id = sys.argv[3]

    run_module(path_to_eval_metrics, path_to_segmentation_metrics, existing_run_id)
    