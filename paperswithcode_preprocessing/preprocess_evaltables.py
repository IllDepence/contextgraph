""" From
        evaluation-tables.json
    generate
        tasks.jsonl  (removes tasks_pre.jsonl)
        models.jsonl
        methods_to_datasets.csv
"""

import csv
import json
import os
from util import name_to_slug

in_dir = '/home/ls3data/datasets/paperswithcode/'
out_dir = '/home/ls3data/datasets/paperswithcode/preprocessed/'

evals_orig_fn = 'evaluation-tables.json'
meths_processed_fn = 'methods.jsonl'
meth_name_to_id = dict()
dsets_processed_fn = 'datasets.jsonl'
dset_name_to_id = dict()
tasks_preprocessed_fn = 'tasks_pre.jsonl'
task_id_to_name = dict()
task_name_to_id = dict()
tasks_new_fn = 'tasks.jsonl'
tasks_new = []
modls_new_fn = 'models.jsonl'
modls_new = dict()
meths_to_dsets_fn = 'methods_to_datasets.csv'
meths_to_dsets = []

with open(os.path.join(out_dir, meths_processed_fn)) as f:
    meth_name_to_id = {
        meth['name']: meth['id']
        for meth in [json.loads(line) for line in f]
    }
with open(os.path.join(out_dir, dsets_processed_fn)) as f:
    dset_name_to_id = {
        dset['name']: dset['id']
        for dset in [json.loads(line) for line in f]
    }
with open(os.path.join(out_dir, tasks_preprocessed_fn)) as f:
    lines = f.readlines()
    task_id_to_name = {
        task['id']: task['name']
        for task in [json.loads(line) for line in lines]
    }
    task_name_to_id = {
        task['name']: task['id']
        for task in [json.loads(line) for line in lines]
    }


with open(os.path.join(in_dir, evals_orig_fn)) as f:
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


eval_tasks = recursively_process_eval_list(evals)

for eval_task in eval_tasks:
    # create task entity
    task_id = task_name_to_id.get(eval_task['name'], None)
    if task_id is None:
        new_slug = name_to_slug(eval_task['name'])
        task_id = 'pwc:task/' + new_slug

    task_new = {
        'id': task_id,
        'name': eval_task['name'],
        'description': eval_task['description'],
        'categories': eval_task['categories']
    }
    tasks_new.append(task_new)

    # process data set and model information
    for eval_dset in eval_task['dsets_tmp']:
        # Can’t rely on dset['lnks_tmp'] to nicely split into
        # task slug and data set slug. Culprit examples:
        # - sota/on-1
        # - sota/zero-shot-transfer-image-classification-on
        # - sota/monocular-cross-view-road-scene-parsing-road-2
        dset_id = dset_name_to_id.get(eval_dset['name'], None)
        is_sub_dset = False
        if dset_id is None:
            dset_id = name_to_slug(eval_dset['name'])
            is_sub_dset = True
        for eval_modl in eval_dset['mdls_tmp']:
            if eval_modl['name'] in meth_name_to_id:
                # model is also treated as a method by PWC
                # -> create method entity anyways
                # -> create model to dataset link
                #    (if not a sub dset
                #     TODO: can we identify the parent dset?
                #           there is a subdataset dict key but
                #           never used in evaluation-tables.json)
                meth_id = meth_name_to_id[eval_modl['name']]
                if not is_sub_dset:
                    meths_to_dsets.append([
                        meth_id,
                        dset_id
                    ])
            modl_id = 'pwc:model/' + name_to_slug(eval_modl['name'])
            modl = modls_new.get(modl_id, None)
            if modl is None:
                # first time we see this. create new entity
                modl = {
                    'id': modl_id,
                    'name': eval_modl['name'],
                    'using_paper_titles': set([eval_modl['paper_title']]),
                    'evaluations': []
                }
                modls_new[modl_id] = modl
            else:
                # we have seen this before. extend
                modls_new[modl_id]['using_paper_titles'].add(
                    eval_modl['paper_title']
                )
            modls_new[modl_id]['evaluations'].append([
                task_id,
                dset_id
            ])


with open(os.path.join(out_dir, tasks_new_fn), 'w') as f:
    for task in tasks_new:
        json.dump(task, f)
        f.write('\n')

with open(os.path.join(out_dir, modls_new_fn), 'w') as f:
    for modl_id, modl in modls_new.items():
        modl['using_paper_titles'] = list(modl['using_paper_titles'])
        json.dump(modl, f)
        f.write('\n')

with open(os.path.join(out_dir, meths_to_dsets_fn), 'w') as f:
    csv_writer = csv.writer(
        f,
        delimiter=',',
        quoting=csv.QUOTE_NONE
    )
    csv_writer.writerow([
        'method_id',
        'dataset_id',
    ])
    for (meth_id, dset_id) in meths_to_dsets:
        csv_writer.writerow([meth_id, dset_id])

os.remove(os.path.join(out_dir, tasks_preprocessed_fn))
