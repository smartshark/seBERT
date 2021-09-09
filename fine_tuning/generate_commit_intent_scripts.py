import sys
import pathlib

basepath = pathlib.Path(__file__)

basepath = str(basepath.parent).replace('scratch1', 'scratch')

tpl = """#!/bin/bash

cd {}

source bin/activate

python3.9 commit_intent_task.py """.format(basepath)

freeze_strategy = sys.argv[1]

for model_name in ['BERTbase', 'BERTlarge', 'BERToverflow', 'seBERT', 'BERTlargenv', 'FastTextAutotuned', 'RandomForest']:
    submits = []
    for run_number in range(10):
        for fold_number in range(10):
            tmp = tpl + '{} {} {} {}'.format(model_name, run_number, fold_number, freeze_strategy)

            nodes = 'rtx5000:1'
            if model_name in ['BERTlarge', 'BERTlargenv']:
                nodes = 'v100:1'

            with open('commit_intent/scripts_{}/{}_{}_{}.sh'.format(freeze_strategy, model_name, run_number, fold_number), 'w') as f:
                f.write(tmp)
            
            submit = 'sbatch -e {5}/commit_intent/logs_{4}/{0}_{1}_{2}.err -o {5}/commit_intent/logs_{4}/{0}_{1}_{2}.out -C scratch --mem=32G -p gpu -G {3} {5}/commit_intent/scripts_{4}/{0}_{1}_{2}.sh'.format(model_name, run_number, fold_number, nodes, freeze_strategy, basepath)
            submits.append(submit)

    with open('commit_intent_submit_{}_{}.sh'.format(freeze_strategy, model_name), 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n'.join(submits))
