import sys
import random

import numpy as np
import torch
import transformers
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, matthews_corrcoef

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from util import BERT, FastTextAutotuned, load_fold

model_name = sys.argv[1]
run_number = int(sys.argv[2])
fold_number = sys.argv[3]
freeze_strategy = sys.argv[4]

random.seed(run_number)
np.random.seed(run_number)
torch.manual_seed(run_number)
torch.cuda.manual_seed(run_number)
torch.cuda.manual_seed_all(run_number)
torch.backends.cudnn.deterministic = True
transformers.set_seed(run_number)

checkpoint_dir = './commit_intent/checkpoints_{}/{}_{}_{}'.format(freeze_strategy, model_name, run_number, fold_number)
output_file = './commit_intent/parts_{}/{}_{}_{}.csv'.format(freeze_strategy, model_name, run_number, fold_number)
batch_size = 8

if 'BERT' in model_name:
    clf = BERT(model_name, checkpoint_dir, freeze_strategy=freeze_strategy, batch_size=batch_size)
if 'FastText' in model_name:
    clf = FastTextAutotuned()
if 'RandomForest' in model_name:
    clf = Pipeline([
        ('vect', CountVectorizer(stop_words='english', strip_accents='ascii')),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier()),
    ])

X_train, y_train, X_test, y_test = load_fold(run_number, fold_number)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

results = [{'model': model_name,
            'run_number': run_number,
            'fold_number': fold_number,
            'mcc': matthews_corrcoef(y_true=y_test, y_pred=y_pred),
            'f1': f1_score(y_true=y_test, y_pred=y_pred),
            'precision': precision_score(y_true=y_test, y_pred=y_pred),
            'recall': recall_score(y_true=y_test, y_pred=y_pred)}]

result_df = pd.DataFrame(results)
result_df.to_csv(output_file, index=False)
