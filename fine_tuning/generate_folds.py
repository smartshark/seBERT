import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

df = pd.read_csv('all_changes_gt.csv')
X = df['message_no_newlines'].values
y = df['internal_quality'].astype(int).values

for run_number in range(10):
    np.random.seed(run_number)
    sp = KFold(n_splits=10, shuffle=True)
    for fold_number, (train_idx, test_idx) in enumerate(sp.split(X, y)):

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        with open('rfolds/{}_{}_X_train_commit_intent.npy'.format(run_number, fold_number), 'wb') as f:
            np.save(f, X_train)
        with open('rfolds/{}_{}_X_test_commit_intent.npy'.format(run_number, fold_number), 'wb') as f:
            np.save(f, X_test)
        with open('rfolds/{}_{}_y_train_commit_intent.npy'.format(run_number, fold_number), 'wb') as f:
            np.save(f, y_train)
        with open('rfolds/{}_{}_y_test_commit_intent.npy'.format(run_number, fold_number), 'wb') as f:
            np.save(f, y_test)
