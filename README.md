# seBERT

This repository contains the code required for the pre-training of seBERT as well as an example for fine-tuning and using the model.
## Create pre-training data
```
bash scripts/prepare_data.sh
```
## Pre-training
1. Build the NVIDIA Docker container enviroment.
```
bash scripts/docker/build.sh
```
2. Launch the container.
```
bash scripts/docker/launch.sh
```
3. From within the container run:
```
bash scripts/run_pretraining_seBERT.sh
```

## Fine-tuning

The complete fine-tuning and evaluation code can be found in the fine\_tuning folder.
