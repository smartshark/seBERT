import pickle
import pandas as pd

train_data_all = pickle.load(open("ReplicationKit/train_data_all.p", "rb"))
test_data_all = pickle.load(open("ReplicationKit/test_data_all.p", "rb"))

dfs = []
for project_name, values in train_data_all.items():
    values['project'] = project_name
    values['text_no_newlines'] = values['title'].str.replace('\n', ' ') + ' ' + values['description'].str.replace('\n', ' ')
    values['classification'] = values['classification'].astype(int)
    values = values.drop(columns=['id', 'title', 'description', 'discussion'])
    dfs.append(values)

for project_name, values in test_data_all.items():
    values['project'] = project_name
    values['text_no_newlines'] = values['title'].str.replace('\n', ' ') + ' ' + values['description'].str.replace('\n', ' ')
    values['classification'] = values['classification'].astype(int)
    values = values.drop(columns=['id', 'title', 'description', 'discussion'])
    dfs.append(values)

df = pd.concat(dfs).reset_index().drop(columns=['index'])
df.to_feather('issue_type_task.feather')
