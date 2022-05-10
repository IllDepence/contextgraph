""" From
        datasets_ext.json (extended w/ using papers through crawling)
    generate
        datasets.jsonl
        datasets_to_papers.jsonl
        datasets_to_tasks.jsonl
"""

import json
import os
from util import url_to_pwc_id

in_dir = '/home/ls3data/datasets/paperswithcode/'
out_dir = '/home/ls3data/datasets/paperswithcode/preprocessed/'

dsets_orig_fn = 'datasets_ext.json'
dsets_orig = []
dsets_new_fn = 'datasets.jsonl'
dsets_new = []
dsets_to_tasks_fn = 'datasets_to_tasks.jsonl'
dsets_to_tasks = []
dsets_to_pprs_fn = 'datasets_to_papers.jsonl'
dsets_to_pprs = []

in_dir = './'  # dev
dsets_orig_fn = 'datasets_ext_dev.json'  # dev

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
    for (dset_id, ppr_id) in dsets_to_pprs:
        json.dump([dset_id, ppr_id], f)
        f.write('\n')

with open(os.path.join(out_dir, dsets_to_tasks_fn), 'w') as f:
    for (dset_id, task_id) in dsets_to_tasks:
        json.dump([dset_id, task_id], f)
        f.write('\n')
