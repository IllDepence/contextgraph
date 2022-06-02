""" From
        datasets_ext.json (extended w/ using papers through crawling)
    generate
        datasets.jsonl
        tasks_pre.jsonl (only id and name, description added from eval tables)
        datasets_to_papers.csv
        datasets_to_tasks.csv
"""

import csv
import json
import os
from contextgraph import config as cg_config
from contextgraph.util import preprocessing as prep_util


def preprocess_datasets():
    dsets_orig = []
    dsets_new_fn = cg_config.graph_dsets_fn
    dsets_new = []
    tasks_pre_fn = cg_config.graph_tasks_pre_fn
    tasks_pre = dict()
    dsets_to_tasks_fn = cg_config.graph_dsets_to_tasks_fn
    dsets_to_tasks = []
    dsets_to_pprs_fn = cg_config.graph_dsets_to_pprs_fn
    dsets_to_pprs = []

    with open(os.path.join(
        cg_config.pwc_data_dir,
        cg_config.pwc_dsets_ext_fn
    )) as f:
        dsets_orig = json.load(f)

    for dset in dsets_orig:
        # create preprocessed dataset object
        dset_new = {
            key: val
            for key, val in dset.items()
            if key not in ['tasks', 'variants', 'using_papers']  # remove
        }
        # add URL slug ID
        dset_id = prep_util.url_to_pwc_id(dset['url'])
        dset_new['id'] = dset_id
        dset_new['type'] = 'dataset'
        dset_new['variant_surface_forms'] = [  # rename
                var_sf for var_sf in dset['variants']
                if var_sf not in [dset['name'], dset['full_name']]
            ]
        # build new dataset list
        dsets_new.append(dset_new)
        # build dataset->using_papers
        for ppr in dset['using_papers']:
            ppr_id = prep_util.url_to_pwc_id(ppr['url'])
            dsets_to_pprs.append(
                [dset_id, ppr_id]
            )
        # build dataset->taks
        for task in dset['tasks']:
            task_id = prep_util.url_to_pwc_id(task['url'])
            tasks_pre[task_id] = {
                'id': task_id,
                'type': 'task',
                'name': task['task']
            }
            dsets_to_tasks.append(
                [dset_id, task_id]
            )

    with open(os.path.join(cg_config.graph_data_dir, dsets_new_fn), 'w') as f:
        for dset in dsets_new:
            json.dump(dset, f)
            f.write('\n')

    with open(os.path.join(cg_config.graph_data_dir, tasks_pre_fn), 'w') as f:
        for task_id, task in tasks_pre.items():
            json.dump(task, f)
            f.write('\n')

    with open(
        os.path.join(cg_config.graph_data_dir, dsets_to_pprs_fn),
        'w'
    ) as f:
        csv_writer = csv.writer(
            f,
            delimiter=',',
            quoting=csv.QUOTE_NONE
        )
        csv_writer.writerow([
            'dataset_id',
            'paper_id',
        ])
        for (dset_id, ppr_id) in dsets_to_pprs:
            csv_writer.writerow([dset_id, ppr_id])

    with open(
        os.path.join(cg_config.graph_data_dir, dsets_to_tasks_fn),
        'w'
    ) as f:
        csv_writer = csv.writer(
            f,
            delimiter=',',
            quoting=csv.QUOTE_NONE
        )
        csv_writer.writerow([
            'dataset_id',
            'task_id',
        ])
        for (dset_id, task_id) in dsets_to_tasks:
            csv_writer.writerow([dset_id, task_id])
