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

## Using seBERT for your own tasks
In this section, we give a small example for fine-tuning the seBERT model and using the fine-tuned version.

We give an example for the fine-tuning process so that you can use the pre-trained seBERT model for your own tasks.
This can be found in the Jupyter Notebook in `notebooks/FineTuning.ipynb`.
In addition, we give an example for the use of the fine-tuned model in a sequence classification task in `notebooks/Classification.ipynb`.

Create a virtualenv within the notebooks dir, install dependencies and download sentiment data.
```bash
cd notebooks
python3.8 -m venv .
source bin/activate
pip install --upgrade pip
pip install torch pandas numpy scikit-learn transformers jupyterlab ipywidgets
cd data
wget -O github_gold.csv https://figshare.com/ndownloader/files/21001260 
cd ../../
```

Load the pre-trained seBERT model.
```bash
cd notebooks/models
wget https://smartshark2.informatik.uni-goettingen.de/sebert/seBERT_pre_trained.tar.gz
tar -xzf seBERT_pre_trained.tar.gz
cd ../../
```

Start Jupyter lab
```bash
cd notebooks
jupyter lab
```

