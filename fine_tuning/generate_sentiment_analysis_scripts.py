import sys
import pathlib

basepath = pathlib.Path(__file__)

basepath = str(basepath.parent).replace('scratch1', 'scratch')

tpl = """#!/bin/bash

cd {}

source bin/activate

python3.9 sentiment_analysis_task.py """.format(basepath)

freeze_strategy = sys.argv[1]
sentiment_part = sys.argv[2]

for model_name in ['BERTbase', 'BERTlarge', 'BERToverflow', 'seBERT', 'BERTlargenv', 'FastTextAutotuned', 'RandomForest']:
    submits = []
    for run_number in range(10):
        for fold_number in range(10):
            tmp = tpl + '{} {} {} {} {}'.format(model_name, run_number, fold_number, freeze_strategy, sentiment_part)

            nodes = 'rtx5000:1'
            if model_name in ['BERTlarge', 'BERTlargenv']:
                nodes = 'v100:1'

            with open('sentiment_analysis/{}/scripts_{}/{}_{}_{}.sh'.format(sentiment_part, freeze_strategy, model_name, run_number, fold_number), 'w') as f:
                f.write(tmp)
            
            submit = 'sbatch -e {5}/sentiment_analysis/{6}/logs_{4}/{0}_{1}_{2}.err -o {5}/sentiment_analysis/{6}/logs_{4}/{0}_{1}_{2}.out -C scratch --mem=32G -p gpu -G {3} {5}/sentiment_analysis/{6}/scripts_{4}/{0}_{1}_{2}.sh'.format(model_name, run_number, fold_number, nodes, freeze_strategy, basepath, sentiment_part)

            if model_name in ['RandomForest', 'FastTextAutotuned']:
                submit = 'sbatch -e {5}/sentiment_analysis/{6}/logs_{4}/{0}_{1}_{2}.err -o {5}/sentiment_analysis/{6}/logs_{4}/{0}_{1}_{2}.out -C scratch --mem=32G {5}/sentiment_analysis/{6}/scripts_{4}/{0}_{1}_{2}.sh'.format(model_name, run_number, fold_number, nodes, freeze_strategy, basepath, sentiment_part)
            submits.append(submit)

    with open('sentiment_analysis_submit_{}_{}_{}.sh'.format(sentiment_part, freeze_strategy, model_name), 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n'.join(submits))
