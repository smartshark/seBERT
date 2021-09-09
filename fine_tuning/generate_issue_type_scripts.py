import sys
import pathlib

basepath = pathlib.Path(__file__)

basepath = str(basepath.parent).replace('scratch1', 'scratch')

PROJECTS = ['ant-ivy', 'archiva', 'calcite', 'cayenne', 'commons-bcel', 'commons-beanutils',
            'commons-codec', 'commons-collections', 'commons-compress', 'commons-configuration',
            'commons-dbcp', 'commons-digester', 'commons-io', 'commons-jcs', 'commons-jexl',
            'commons-lang', 'commons-math', 'commons-net', 'commons-scxml',
            'commons-validator', 'commons-vfs', 'deltaspike', 'eagle', 'giraph', 'gora', 'jspwiki',
            'knox', 'kylin', 'lens', 'mahout', 'manifoldcf','nutch','opennlp','parquet-mr',
            'santuario-java', 'systemml', 'tika', 'wss4j', 'httpcomponents-client', 'jackrabbit', 'rhino', 'tomcat', 'lucene-solr']


tpl = """#!/bin/bash

cd {}

source bin/activate

python3.9 issue_type_task.py """.format(basepath)

freeze_strategy = sys.argv[1]

for model_name in ['BERTbase', 'BERTlarge', 'BERToverflow', 'seBERT', 'BERTlargenv', 'RandomForest', 'FastTextAutotuned', 'seBERTfinal']:
    submits = []
    for project_name in PROJECTS:
        tmp = tpl + '{} {} {}'.format(model_name, project_name, freeze_strategy)

        nodes = 'rtx5000:1'
        if model_name in ['BERTlarge', 'BERTlargenv'] and freeze_strategy == 'no_freeze':
            nodes = 'v100:1'    

        with open('issue_type/scripts_{}/{}_{}.sh'.format(freeze_strategy, model_name, project_name), 'w') as f:
            f.write(tmp)
        
        submit = 'sbatch -e {4}/issue_type/logs_{3}/{0}_{1}.err -o {4}/issue_type/logs_{3}/{0}_{1}.out -C scratch --mem=32G -p gpu -G {2} -t 2-00:00:00 {4}/issue_type/scripts_{3}/{0}_{1}.sh'.format(model_name, project_name, nodes, freeze_strategy, basepath)
        submits.append(submit)

    with open('issue_type_submit_{}_{}.sh'.format(freeze_strategy, model_name), 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n'.join(submits))

