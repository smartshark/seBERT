# Replication kit seBERT fine-tuning evaluation

This part of the replication kit starts with pre-trained models.
The goal is to execute fine-tuning for different pre-trained models and tasks.

## Create venv and install dependencies

```bash
python3 -m venv .
source bin/activate
pip install -r requirements.txt
```

## Download pre-trained models

BERTbase and BERTlarge are automatically fetched via the huggingface library. For BERToverflow and seBERT we need to fetch them manually.

### BERToverflow

The BERToverflow data is only available on a google drive which can be found [here](https://github.com/lanwuwei/BERTOverflow) repository.
After downloading the required files (vocab, config and checkpoint data) the model must be converted into a pytorch model as the scripts expect this.

The Huggingface library provides a tool for this which takes the checkpoint and bert\_config.
If you have downloaded everything into the main directory of this replication kit you can convert the model like this.

```bash
transformers-cli convert --model_type bert --tf_checkpoint model.ckpt-1000000 --config bert_config.json --pytorch_dump_output ./models/BERToverflow/pytorch_model.bin

cp bert_config.json ./models/BERToverflow/config.json
cp vocab.txt ./models/BERToverflow/
```

### seBERT

```bash
cd models
wget https://smartshark2.informatik.uni-goettingen.de/seBERT/seBERT.tar.gz
tar -xzf seBERT.tar.gz
```

## Evaluate fine-tuning

To evaluate seBERT for software engineering research tasks we use the pre-trained model and fine-tune it for three tasks. The results are written as a CSV file per run.

### Commit intent prediction task

The goal of this task is to predict if the intent of the commit is to improve the general quality, i.e., clean up, simplify code, refactor.
The data consists of 2533 commits manually labeled by two researchers.
The commit message classification task contains 10 repetitions for every fold of a 10-fold cross-validation.
This means that we have 100 separate fine-tuning tasks. To ensure that everyone gets the same data we split our data beforehand.

#### Prepare data

```bash
wget https://zenodo.org/record/5494134/files/all_changes_gt.csv.gz
gunzip all_changes_gt.csv.gz
python generate_folds.py
```

#### Execute fine-tuning task

We are using  a HPC system with GPU/TPU queues.
In order to make use of the parallelization the jobs are split up.
To easily submit the larger number of jobs, we create sh scripts for submission 
to the HPC system.

```bash
python generate_commit_intent_scripts.py no_freeze
```

With no\_freeze we allow the fine-tuning of all weights in the BERT models.
Instead of an HPC system with multiple jobs this can be handled on a single machine.
Just take a look into utils.py and commit\_intent\_task.py.


### Issue type prediction task

The goal of this task is to predict whether the type of the issue is bug or other, e.g., feature request or improvement. 
The data consists of manually labeled issue types by three researchers.


#### Prepare data

Download the replication kit from [zenodo](https://zenodo.org/record/3994254/) and convert the data to be used
in the task.

```bash
wget https://zenodo.org/record/3994254/files/replication-kit-issue-type-prediction.zip
unzip replication-kit-issue-type-prediction.zip

python prepare_issue_type_data.py
```

#### Execute fine-tuning task

Generate execution scripts for the HPC system.

```bash
python generate_issue_type_scripts.py no_freeze
```


### Sentiment Analysis task

In this task, we perform sentiment analysis on three datasets.

#### Prepare data

Download [Github comments](https://doi.org/10.6084/m9.figshare.11604597), [API reviews](https://github.com/giasuddin/OpinionValueTSE/blob/master/BenchmarkUddinSO-ConsoliatedAspectSentiment.xls) and [Jira/App/SO data](https://sentiment-se.github.io/replication.zip) to ./sentiment\_analysis/data/.

```bash
python prepare_sentiment_data.py
```


#### Execute fine-tuning task

Generate execution scripts for the HPC system.

```bash
python generate_sentiment_scripts.py no_freeze
```

## Results

After running the fine-tuning the results should be available in the parts folders.
For every run one CSV file is generated with common model performance metrics.
