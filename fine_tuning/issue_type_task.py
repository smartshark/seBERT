import sys
import random

import numpy as np
import torch
import pandas as pd
import transformers

from sklearn.metrics import recall_score, precision_score, f1_score, matthews_corrcoef

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from util import BERT, FastTextAutotuned

PROJECTS = ['ant-ivy', 'archiva', 'calcite', 'cayenne', 'commons-bcel', 'commons-beanutils',
            'commons-codec', 'commons-collections', 'commons-compress', 'commons-configuration',
            'commons-dbcp', 'commons-digester', 'commons-io', 'commons-jcs', 'commons-jexl',
            'commons-lang', 'commons-math', 'commons-net', 'commons-scxml',
            'commons-validator', 'commons-vfs', 'deltaspike', 'eagle', 'giraph', 'gora', 'jspwiki',
            'knox', 'kylin', 'lens', 'mahout', 'manifoldcf','nutch','opennlp','parquet-mr',
            'santuario-java', 'systemml', 'tika', 'wss4j', 'httpcomponents-client', 'jackrabbit', 'rhino', 'tomcat', 'lucene-solr']

df = pd.read_feather('issue_type_task.feather')

model_name = sys.argv[1]
test_project = sys.argv[2]
freeze_strategy = sys.argv[3]

seed_number = PROJECTS.index(test_project)

random.seed(seed_number)
np.random.seed(seed_number)
torch.manual_seed(seed_number)
torch.cuda.manual_seed(seed_number)
torch.cuda.manual_seed_all(seed_number)
torch.backends.cudnn.deterministic = True
transformers.set_seed(seed_number)

checkpoint_dir = './issue_type/checkpoints_{}/{}_{}'.format(freeze_strategy, model_name, test_project)
output_file = './issue_type/parts_{}/{}_{}.csv'.format(freeze_strategy, model_name, test_project)
batch_size = 8

if 'BERT' in model_name:
    #if freeze_strategy == 'no_freeze' and model_name == 'BERTlarge':
    #    batch_size = 8
    clf = BERT(model_name, checkpoint_dir, freeze_strategy=freeze_strategy, batch_size=batch_size)
if 'FastText' in model_name:
    clf = FastTextAutotuned()
if 'RandomForest' in model_name:
    clf = Pipeline([
        ('vect', CountVectorizer(stop_words='english', strip_accents='ascii')),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier()),
    ])

X_test = df[df['project'] == test_project]['text_no_newlines'].values
X_train = df[df['project'] != test_project]['text_no_newlines'].values
y_test = df[df['project'] == test_project]['classification'].values.astype(int)
y_train = df[df['project'] != test_project]['classification'].values.astype(int)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

results = [{'model': model_name,
            'test_project': test_project,
            'mcc': matthews_corrcoef(y_true=y_test, y_pred=y_pred),
            'f1': f1_score(y_true=y_test, y_pred=y_pred),
            'precision': precision_score(y_true=y_test, y_pred=y_pred),
            'recall': recall_score(y_true=y_test, y_pred=y_pred)}]

result_df = pd.DataFrame(results)
result_df.to_csv(output_file, index=False)
