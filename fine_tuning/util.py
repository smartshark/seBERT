import os
import gc
import threading
import sys

import numpy as np
import torch

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, KFold

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer

from skift.util import temp_dataset_fpath, dump_xy_to_fasttext_format
from skift import FirstColFtClassifier


def compute_metrics_multi_label(p):
    """This metrics computation is only for finetuning."""
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='micro')
    precision = precision_score(y_true=labels, y_pred=pred, average='micro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='micro')
    mcc = matthews_corrcoef(y_true=labels, y_pred=pred)
    
    recall_ma = recall_score(y_true=labels, y_pred=pred, average='macro')
    precision_ma = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1_ma = f1_score(y_true=labels, y_pred=pred, average='macro')

    return {'accuracy': accuracy, 'precision_micro': precision, 'recall_micro': recall, 'f1_micro': f1, 'mcc': mcc, 'precision_macro': precision_ma, 'recall_macro': recall_ma, 'f1_macro': f1_ma}


def compute_metrics(p):
    """This metrics computation is only for finetuning."""
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    mcc = matthews_corrcoef(y_true=labels, y_pred=pred)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'mcc': mcc}

def load_sentiment_label(run_number, fold_number, sentiment_part):
    """Load pre-generated data for run and fold numbers in case of sentiment classification."""
    with open('./sentiment_analysis/data/rfolds/{}_{}_X_train_{}.npy'.format(run_number, fold_number, sentiment_part), 'rb') as f:
        X_train = np.load(f, allow_pickle=True)
    with open('./sentiment_analysis/data/rfolds/{}_{}_X_test_{}.npy'.format(run_number, fold_number, sentiment_part), 'rb') as f:
        X_test = np.load(f, allow_pickle=True)
    with open('./sentiment_analysis/data/rfolds/{}_{}_y_train_{}.npy'.format(run_number, fold_number, sentiment_part), 'rb') as f:
        y_train = np.load(f, allow_pickle=True)
    with open('./sentiment_analysis/data/rfolds/{}_{}_y_test_{}.npy'.format(run_number, fold_number, sentiment_part), 'rb') as f:
        y_test = np.load(f, allow_pickle=True)
    return X_train, y_train, X_test, y_test

def load_fold_multi_label(run_number, fold_number):
    """Load pre-generated data for run and fold numbers in case of multi-label classification."""
    with open('ml_rfolds/{}_{}_X_train_commit_intent.npy'.format(run_number, fold_number), 'rb') as f:
        X_train = np.load(f, allow_pickle=True)
    with open('ml_rfolds/{}_{}_X_test_commit_intent.npy'.format(run_number, fold_number), 'rb') as f:
        X_test = np.load(f, allow_pickle=True)
    with open('ml_rfolds/{}_{}_y_train_commit_intent.npy'.format(run_number, fold_number), 'rb') as f:
        y_train = np.load(f, allow_pickle=True)
    with open('ml_rfolds/{}_{}_y_test_commit_intent.npy'.format(run_number, fold_number), 'rb') as f:
        y_test = np.load(f, allow_pickle=True)
    return X_train, y_train, X_test, y_test

def load_fold(run_number, fold_number):
    """Load pre-generated data for run and fold numbers."""
    with open('rfolds/{}_{}_X_train_commit_intent.npy'.format(run_number, fold_number), 'rb') as f:
        X_train = np.load(f, allow_pickle=True)
    with open('rfolds/{}_{}_X_test_commit_intent.npy'.format(run_number, fold_number), 'rb') as f:
        X_test = np.load(f, allow_pickle=True)
    with open('rfolds/{}_{}_y_train_commit_intent.npy'.format(run_number, fold_number), 'rb') as f:
        y_train = np.load(f, allow_pickle=True)
    with open('rfolds/{}_{}_y_test_commit_intent.npy'.format(run_number, fold_number), 'rb') as f:
        y_test = np.load(f, allow_pickle=True)
    return X_train, y_train, X_test, y_test


def get_model_and_tokenizer(model_name, num_labels=2):
    """Load model_name and tokenizer, assumes paths from the Readme"""
    model = None
    tokenizer = None
    if model_name == 'seBERT':
        model = BertForSequenceClassification.from_pretrained('./models/seBERT/pytorch_model.bin', config='./models/seBERT/config.json', num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained('./models/seBERT/', do_lower_case=True)
    elif model_name == 'BERToverflow':
        model = BertForSequenceClassification.from_pretrained('./models/BERToverflow/pytorch_model.bin', config='./models/BERToverflow/config.json', num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained('./models/BERToverflow/')
    elif model_name == 'BERTbase':
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif model_name == 'BERTlarge':
        model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    elif model_name == 'BERTlargenv':
        model = BertForSequenceClassification.from_pretrained('./models/BERTlargenv/pytorch_model.bin', config='./models/BERTlargenv/config.json', num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained('./models/BERTlargenv/')
    return model, tokenizer

def cv_autotune_fasttext_fit(X, y, *, cv_folds, autotuneDuration=300):
    """
    Helper method to train fastText with autotuning through cross valdiation. This is not yet supported by the API.
    :param X: features
    :param y: labels
    :param input_col_lbl: input column that is used from features
    :param cv_folds: number of cv folds
    :param autotuneDuration: duration of the autotune (Default: 300)
    :return: fitted autotuned fastText classifier
    """
    train_test_folds = []
    cv_clfs = []
    cv_args = []
    folds = KFold(n_splits=cv_folds, shuffle=True)
    for train_index, test_index in folds.split(X, y):
        train_test_folds.append((train_index, test_index))

        tmp_file = temp_dataset_fpath()
        dump_xy_to_fasttext_format(X[test_index], y[test_index], tmp_file)

        # redirect stdout from C++ code into variable, we get the hyper parameters from there
        stdout_fileno = sys.stdout.fileno()
        stdout_save = os.dup(stdout_fileno)
        stdout_pipe = os.pipe()
        os.dup2(stdout_pipe[1], stdout_fileno)
        os.close(stdout_pipe[1])
        captured_stdout = b''

        # use a thread to drain the pipe to avoid a potential deadlock
        def drain_pipe():
            nonlocal captured_stdout
            while True:
                data = os.read(stdout_pipe[0], 1024)
                if not data:
                    break
                captured_stdout += data
        t = threading.Thread(target=drain_pipe)
        t.start()

        # we use the skift-wrapper for the fasttext api
        # training must be with verbose>2 to trigger the output of the hyper parameters
        cv_clf = FirstColFtClassifier(autotuneValidationFile=tmp_file, verbose=3, autotuneDuration=autotuneDuration)
        cv_clf.fit(X, y)

        # stop redirection of stdout and close pipe
        os.close(stdout_fileno)
        t.join()
        os.close(stdout_pipe[0])
        os.dup2(stdout_save, stdout_fileno)
        os.close(stdout_save)

        # parse best params from output stream and prepare them as kwargs-dict
        captured_out = captured_stdout.decode(sys.stdout.encoding)
        cur_cv_args = {}
        for bestarg in captured_out.split('Best selected args')[-1].strip().split('\n')[1:]:
            arg = bestarg.split('=')[0].strip()
            val = bestarg.split('=')[-1].strip()
            if arg not in 'dsub':
                try:
                    cur_cv_args[arg] = int(val)
                except ValueError:
                    try:
                        cur_cv_args[arg] = float(val)
                    except ValueError:
                        cur_cv_args[arg] = val

        # cleanup validation file
        try:
            os.remove(tmp_file)
        except FileNotFoundError:
            pass

        cv_clfs.append(cv_clf)
        cv_args.append(cur_cv_args)

    best_args = None
    best_score = None
    for i, (train_index, test_index) in enumerate(train_test_folds):

        y_pred = np.around(cv_clfs[i].predict_proba(X[test_index])[:, 1], decimals=0)
        cur_score = f1_score(y[test_index], y_pred, average='macro')
        if best_score is None or best_score < cur_score:
            best_args = cv_args[i]
            best_score = cur_score
    print('Best args of cv-autotune: %s' % best_args)
    # fit classifier with best arguments on all data
    clf = FirstColFtClassifier(**best_args)
    clf.fit(X, y)
    return clf


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


class BERT(BaseEstimator, ClassifierMixin):
    def __init__(self, model_name, checkpoints_dir, freeze_strategy=None, batch_size=8, num_labels=2):
        self.model_name = model_name
        self.trainer = None
        self.checkpoints_dir = checkpoints_dir
        self.last_layer = 'layer.23'
        self.max_length = 512
        self.model, self.tokenizer = get_model_and_tokenizer(model_name, num_labels=num_labels)
        self.freeze_strategy = freeze_strategy
        self.batch_size = batch_size
        self.num_labels = num_labels
        if model_name == 'seBERT':
            self.max_length = 128
        if model_name == 'seBERTfinal':
            self.max_length = 128
        if model_name == 'BERTbase':
            self.last_layer = 'layer.11'

    def fit(self, X, y):
        """Fit is finetuning from the pre-trained model."""

        if self.freeze_strategy == 'last2layer':
            for name, param in self.model.bert.named_parameters():
                if not name.startswith('pooler') and self.last_layer not in name:
                    param.requires_grad = False

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        X_train_tokens = self.tokenizer(X_train.tolist(), padding=True, truncation=True, max_length=self.max_length)
        X_val_tokens = self.tokenizer(X_val.tolist(), padding=True, truncation=True, max_length=self.max_length)

        train_dataset = Dataset(X_train_tokens, y_train)
        eval_dataset = Dataset(X_val_tokens, y_val)

        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        training_args = TrainingArguments(
            output_dir                  = self.checkpoints_dir,
            num_train_epochs            = 5,
            per_device_train_batch_size = self.batch_size,
            per_device_eval_batch_size  = self.batch_size,
            gradient_accumulation_steps = 4,
            eval_accumulation_steps     = 10,
            evaluation_strategy         = 'epoch', 
            load_best_model_at_end      = True
        )
        #if self.model_name == 'BERTlarge':
        #    training_args.fp16 = True
        #    training_args.deepspeed = './ds_config.json'

        self.trainer = Trainer(
            model           = self.model,
            args            = training_args,
            train_dataset   = train_dataset,
            eval_dataset    = eval_dataset,
            compute_metrics = compute_metrics
        )
        if self.num_labels > 2:
            self.trainer.compute_metrics = compute_metrics_multi_label
        
        print(self.trainer.train())
        return self

    def predict_proba(self, X, y=None):
        """This is kept simple intentionally, for larger Datasets this would be too ineficient,
        because we would effectively have a batch size of 1."""
        y_probs = []
        self.trainer.model.eval()
        with torch.no_grad():
            for _, X_row in enumerate(X):
                inputs = self.tokenizer(X_row, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to('cuda')
                outputs = self.trainer.model(**inputs)
                probs = outputs[0].softmax(1).cpu().detach().numpy()
                y_probs.append(probs)
        return y_probs

    def predict(self, X, y=None):
        """Predict is evaluation."""
        y_probs = self.predict_proba(X, y)
        y_pred = []
        for y_prob in y_probs:
            if self.num_labels > 2:
                y_pred.append(y_prob.argmax())
            else:
                y_pred.append(y_prob.argmax() == 1)
        return y_pred

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.trainer.model.save_pretrained(path)


class FastTextAutotuned(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.threshold = 0.5
        self.cv_folds = 3
        self.autotune_duration = 90
        self.sk_clf = None

    def _transform_X(self, X):
        z = np.zeros(len(X))
        return np.c_[X, z]

    def fit(self, X, y):
        X_sk = self._transform_X(X)
        self.sk_clf = cv_autotune_fasttext_fit(X_sk, y, cv_folds=self.cv_folds, autotuneDuration=self.autotune_duration)
        return self

    def predict_proba(self, X, y=None):
        X_sk = self._transform_X(X)
        return self.sk_clf.predict_proba(X_sk)

    def predict(self, X, y=None):
        X_sk = self._transform_X(X)
        return np.around(self.sk_clf.predict_proba(X_sk)[:, 1], decimals=0)
