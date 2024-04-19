This code was used to obtain the results presented in the paper "An Argumentation Scheme-based Framework for Automatic Reconstruction of Natural Language Enthymemes" (Z Delas, B Pluss, R Ruiz-Dolz) submitted to COMMA 2024.

The code is written in components, each of which can be run independently, or chained together to form a pipeline. \
Each component outputs its data into a subfolder whose name is taken from the last CLI parameter. \
For example, to evaluate the baseline/benchmark model, run benchmark-evaluate, providing it with paths to data from the previous component:
``` 
python components/benchmark-evaluate/evaluate-stage-2-model.py components/benchmark-pair-up/outputs/pipeline_3_no_conclusions/data/pairs-from-all-schemes.json components/benchmark-train/outputs/benchmark_1/models benchmark_3_no_conclusions 
```

All requirements, bar Python 3, are provided in requirements.txt file. An NVIDIA/CUDA platform is assumed.
