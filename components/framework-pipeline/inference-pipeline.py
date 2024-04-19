import os
import sys
from datetime import datetime
import uuid

label_count = 20

if(sys.prefix == sys.base_prefix):
    print("Warning: Virtual env not active")
if(os.getenv("CUDA_VISIBLE_DEVICES") == None):
    print("Warning: CUDA GPU index not specified")

s2_model_run_id = 'run_5-segment-by-argument'
pipeline_run_id = 'pipeline_2_no_conclusions'
print("===== PIPELINE: S2 inference models")
for arg_scheme in range(label_count):
    print(f"===== PIPELINE: S2 inference models (scheme {arg_scheme})")
    os.system(f'python inference-pipeline/s2-inference/s2.py modules/train-s2/outputs/{s2_model_run_id}/models/stage-2-scheme-{arg_scheme} inference-pipeline/pair-up-and-separate/outputs/pipeline_2_no_conclusions/data/scheme-{arg_scheme}-pairs.json {arg_scheme} {pipeline_run_id}')
