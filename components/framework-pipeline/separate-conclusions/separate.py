import os, json, sys
from pathlib import Path

def abs_path(relative_path):
    base_path = Path(__file__).parent
    return (base_path / relative_path).resolve()

def run_module(premises_path, run_id):
    with open(premises_path) as filehandle:
        premises = json.load(filehandle)
        dev_premises = premises['dev']

    dev_premises_without_conclusions = [premise for premise in dev_premises if not premise["is conclusion"]]
    dev_conclusions_only = [premise for premise in dev_premises if premise["is conclusion"]]

    outputs_path = abs_path(f'outputs/{run_id}')
    os.makedirs(f'{outputs_path}/data', exist_ok=True)
    with open(f'{outputs_path}/data/dev-premises-without-conclusions.json', "w") as file:
        json.dump(dev_premises_without_conclusions, file, indent=2)
    with open(f'{outputs_path}/data/dev-conclusions-only.json', "w") as file:
        json.dump(dev_conclusions_only, file, indent=2)

    input_metrics = {
        "dev_premises": len(dev_premises),
        "premises without conclusions": len(dev_premises_without_conclusions),
        "conclusions only:": len(dev_conclusions_only)
    }

    os.makedirs(f'{outputs_path}/metrics', exist_ok=True)
    with open(f'{outputs_path}/metrics/dev-split.json', "w") as file:
        json.dump(input_metrics, file, indent=2)

if __name__ == "__main__":
    premises_path = sys.argv[1]
    run_id = sys.argv[2]

    run_module(premises_path, run_id)

# python inference-pipeline/separate-conclusions/separate.py \
# modules/extract-into-premises/outputs/run_5-segment-by-argument/data/premises.json \
# pipeline_1