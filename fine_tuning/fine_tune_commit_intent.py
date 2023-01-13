import pandas as pd

from util import BERT

model_name = 'seBERT'
freeze_strategy = 'no_freeze'

checkpoint_dir = './commit_intent/seBERT_fine_tuned/'
batch_size = 16

clf = BERT(model_name, checkpoint_dir, freeze_strategy=freeze_strategy, batch_size=batch_size)

df = pd.read_csv('commit_intent.csv.gz')
X = df['message_no_newlines'].values
y = df['internal_quality'].values

clf.fit(X, y)
clf.save_model('./commit_intent/model/')
