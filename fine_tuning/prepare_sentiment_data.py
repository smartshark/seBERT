import pandas as pd

so = pd.read_csv('./sentiment_analysis/data/dataset/StackOverflow.csv')
gh = pd.read_csv('./sentiment_analysis/data/github_gold.csv', delimiter=';')
api = pd.read_excel('./sentiment_analysis/data/BenchmarkUddinSO-ConsoliatedAspectSentiment.xls')

def polarity(row):
    if row['Polarity'] == 'negative':
        return 0
    if row['Polarity'] == 'neutral':
        return 1
    if row['Polarity'] == 'positive':
        return 2

def manual_label(row):
    if row['ManualLabel'] == 'n':
        return 0
    if row['ManualLabel'] == 'o':
        return 1
    if row['ManualLabel'] == 'p':
        return 2

tgh = gh[['Text', 'Polarity']].copy()
tgh['text'] = tgh['Text'].str.replace('\n', ' ')
tgh['sentiment'] = tgh.apply(polarity, axis=1)
tgh = tgh[['text', 'sentiment']].copy()

tso = so[['text', 'oracle']].copy()
tso.rename(columns={'oracle': 'sentiment'}, inplace=True)

tapi = api[['sent', 'ManualLabel']].copy()
tapi['sentiment'] = tapi.apply(manual_label, axis=1)
tapi.rename(columns={'sent': 'text'}, inplace=True)
tapi = tapi[['text', 'sentiment']].copy()

tso.to_csv('./sentiment_analysis/data/so.csv.gz', index=False)
tapi.to_csv('./sentiment_analysis/data/api.csv.gz', index=False)
tgh.to_csv('./sentiment_analysis/data/github.csv.gz', index=False)
