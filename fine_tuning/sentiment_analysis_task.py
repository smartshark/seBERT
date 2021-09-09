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

from util import BERT, FastTextAutotuned, load_sentiment_label

model_name = sys.argv[1]
run_number = int(sys.argv[2])
fold_number = sys.argv[3]
freeze_strategy = sys.argv[4]
sentiment_part = sys.argv[5]

random.seed(run_number)
np.random.seed(run_number)
torch.manual_seed(run_number)
torch.cuda.manual_seed(run_number)
torch.cuda.manual_seed_all(run_number)
torch.backends.cudnn.deterministic = True
transformers.set_seed(run_number)

checkpoint_dir = './sentiment_analysis/{}/checkpoints_{}/{}_{}_{}'.format(sentiment_part, freeze_strategy, model_name, run_number, fold_number)
output_file = './sentiment_analysis/{}/parts_{}/{}_{}_{}.csv'.format(sentiment_part, freeze_strategy, model_name, run_number, fold_number)
batch_size = 8

if 'BERT' in model_name:
    clf = BERT(model_name, checkpoint_dir, freeze_strategy=freeze_strategy, batch_size=batch_size, num_labels=3)
if 'FastText' in model_name:
    clf = FastTextAutotuned()
if 'RandomForest' in model_name:
    clf = Pipeline([
        ('vect', CountVectorizer(stop_words='english', strip_accents='ascii')),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier()),
    ])

X_train, y_train, X_test, y_test = load_sentiment_label(run_number, fold_number, sentiment_part)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

results = [{'model': model_name,
            'run_number': run_number,
            'fold_number': fold_number,
            'mcc': matthews_corrcoef(y_true=y_test, y_pred=y_pred),
            'micro_f1': f1_score(y_true=y_test, y_pred=y_pred, average='micro'),
            'micro_precision': precision_score(y_true=y_test, y_pred=y_pred, average='micro'),
            'micro_recall': recall_score(y_true=y_test, y_pred=y_pred, average='micro'),
            'macro_f1': f1_score(y_true=y_test, y_pred=y_pred, average='macro'),
            'macro_precision': precision_score(y_true=y_test, y_pred=y_pred, average='macro'),
            'macro_recall': recall_score(y_true=y_test, y_pred=y_pred, average='macro')}]

result_df = pd.DataFrame(results)
result_df.to_csv(output_file, index=False)
