import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

for name, df in [('github', pd.read_csv('./sentiment_analysis/data/github.csv.gz')), ('so', pd.read_csv('./sentiment_analysis/data/so.csv.gz')), ('api', pd.read_csv('./sentiment_analysis/data/api.csv.gz'))]:
    X = df['text'].values
    y = df['sentiment'].values


    for run_number in range(10):
        np.random.seed(run_number)
        sp = KFold(n_splits=10, shuffle=True, random_state=run_number)
        for fold_number, (train_idx, test_idx) in enumerate(sp.split(X, y)):

            X_train = X[train_idx]
            y_train = y[train_idx]

            X_test = X[test_idx]
            y_test = y[test_idx]

            with open('./sentiment_analysis/data/rfolds/{}_{}_X_train_{}.npy'.format(run_number, fold_number, name), 'wb') as f:
                np.save(f, X_train)
            with open('./sentiment_analysis/data/rfolds/{}_{}_X_test_{}.npy'.format(run_number, fold_number, name), 'wb') as f:
                np.save(f, X_test)
            with open('./sentiment_analysis/data/rfolds/{}_{}_y_train_{}.npy'.format(run_number, fold_number, name), 'wb') as f:
                np.save(f, y_train)
            with open('./sentiment_analysis/data/rfolds/{}_{}_y_test_{}.npy'.format(run_number, fold_number, name), 'wb') as f:
                np.save(f, y_test)
