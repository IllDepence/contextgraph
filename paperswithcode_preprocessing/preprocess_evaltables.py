""" From
        evaluation-tables.json
    generate
        ...
"""

import json
import os
from util import url_to_pwc_id

in_dir = '/home/ls3data/datasets/paperswithcode/'
out_dir = '/home/ls3data/datasets/paperswithcode/preprocessed/'

evals_fn = 'evaluation-tables.json'
tasks_new_fn = 'tasks.jsonl'
tasks_new = []
modls_new_fn = 'datasets_to_tasks.jsonl'
modls_new = []
subdsets = []

with open(os.path.join(in_dir, evals_fn)) as f:
    evals = json.load(f)

# extract
# - task descriptions
# - task hierarchy
# - task categories
# - models


def recursively_process_eval_list(evals):
    tasks = []
    for evl in evals:
        tasks += recursively_process_eval_list(evl['subtasks'])
        task = {
            'name': evl['task'],
            'description': evl['description'],
            'categories': evl['categories'],
            'dsets_tmp': []
        }
        for dset in evl['datasets']:
            dset_tmp = {
                'name': dset['dataset'],
                'lnks_tmp': [l['url'] for l in dset['dataset_links']],
                'mdls_tmp': []
            }
            for sota_row in dset['sota']['rows']:
                mdl_tmp = {
                    'name': sota_row['model_name'],
                    'paper_title': sota_row['paper_title']
                }
                dset_tmp['mdls_tmp'].append(mdl_tmp)
            task['dsets_tmp'].append(dset_tmp)
        tasks.append(task)
    return tasks


tasks_pre = recursively_process_eval_list(evals)
# TODO
# take model and dataset info out and create separate lists and mappings

# import pprint
# pprint.pprint(tasks_pre)
