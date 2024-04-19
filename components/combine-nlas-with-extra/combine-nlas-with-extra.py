import sys, json, os
from pathlib import Path

def abs_path(relative_path):
    base_path = Path(__file__).parent
    return (base_path / relative_path).resolve()

def run_module(nlas_path, nlas_extra_path, labels_map_path, run_id):
    with open(nlas_path, "r", encoding="utf-8") as read_file:
        nlas = json.load(read_file)
    with open(nlas_extra_path, "r") as file:
        nlas_extra = json.load(file)
    with open(labels_map_path, "r") as file:
        labels_map = json.load(file)

    all_arguments = []
    for k,v in nlas.items():
        v["argument id"] = int(k)
        v["is nlas extra"] = False
        v["argumentation scheme label"] = labels_map[v["argumentation scheme"]]
        all_arguments.append(v)

    for k,v in nlas_extra.items():
        v["argument id"] = int(k) + 2000
        v["is nlas extra"] = True
        v["argumentation scheme label"] = labels_map[v["argumentation scheme"]]
        all_arguments.append(v)

    outputs_path = abs_path(f'outputs/{run_id}')
    os.makedirs(f'{outputs_path}/data', exist_ok=True)
    with open(f"{outputs_path}/data/merged_nlas_with_extra.json", "w") as file:
        json.dump(all_arguments, file, indent=2)

if __name__ == "__main__":
    nlas_path = sys.argv[1]
    nlas_extra_path = sys.argv[2]
    labels_map_path = sys.argv[3]
    run_id = sys.argv[4]

    run_module(nlas_path, nlas_extra_path, labels_map_path, run_id)
