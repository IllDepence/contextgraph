""" From
        datasets_ext.json (extended w/ using papers through crawling)
    generate
        datasets.jsonl
        datasets_to_papers.csv
        datasets_to_tasks.csv
"""

import csv
import json
import os
from util import url_to_pwc_id

in_dir = '/home/ls3data/datasets/paperswithcode/'
out_dir = '/home/ls3data/datasets/paperswithcode/preprocessed/'

dsets_orig_fn = 'datasets_ext.json'
dsets_orig = []
dsets_new_fn = 'datasets.jsonl'
dsets_new = []
dsets_to_tasks_fn = 'datasets_to_tasks.csv'
dsets_to_tasks = []
dsets_to_pprs_fn = 'datasets_to_papers.csv'
dsets_to_pprs = []

with open(os.path.join(in_dir, dsets_orig_fn)) as f:
    dsets_orig = json.load(f)

for dset in dsets_orig:
    # create preprocessed dataset object
    dset_new = {
        key: val
        for key, val in dset.items()
        if key not in ['tasks', 'variants', 'using_papers']  # remove
    }
    # add URL slug ID
    dset_id = url_to_pwc_id(dset['url'])
    dset_new['id'] = dset_id
    dset_new['variant_surface_forms'] = [  # rename
            var_sf for var_sf in dset['variants']
            if var_sf not in [dset['name'], dset['full_name']]
        ]
    # build new dataset list
    dsets_new.append(dset_new)
    # build dataset->using_papers
    for ppr in dset['using_papers']:
        ppr_id = url_to_pwc_id(ppr['url'])
        dsets_to_pprs.append(
            [dset_id, ppr_id]
        )
    # build dataset->taks
    for task in dset['tasks']:
        task_id = url_to_pwc_id(task['url'])
        dsets_to_tasks.append(
            [dset_id, task_id]
        )

with open(os.path.join(out_dir, dsets_new_fn), 'w') as f:
    for dset in dsets_new:
        json.dump(dset, f)
        f.write('\n')

with open(os.path.join(out_dir, dsets_to_pprs_fn), 'w') as f:
    csv_writer = csv.writer(
        f,
        delimiter=',',
        quoting=csv.QUOTE_NONE)
    csv_writer.writerow([
        'dataset_id',
        'paper_id',
    ])
    for (dset_id, ppr_id) in dsets_to_pprs:
        csv_writer.writerow([dset_id, ppr_id])

with open(os.path.join(out_dir, dsets_to_tasks_fn), 'w') as f:
    csv_writer = csv.writer(
        f,
        delimiter=',',
        quoting=csv.QUOTE_NONE)
    csv_writer.writerow([
        'dataset_id',
        'task_id',
    ])
    for (dset_id, task_id) in dsets_to_tasks:
        csv_writer.writerow([dset_id, ppr_id])
