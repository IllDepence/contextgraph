""" From
        papers.json
    generate
        papers.jsonl
        tasks_to_papers.csv
        methods_to_papers.csv
"""

import csv
import json
import os
from util import url_to_pwc_id

in_dir = '/home/ls3data/datasets/paperswithcode/'
out_dir = '/home/ls3data/datasets/paperswithcode/preprocessed/'

pprs_orig_fn = 'papers-with-abstracts.json'
pprs_orig = []
pprs_new_fn = 'papers.jsonl'
pprs_new = []
meths_to_pprs_fn = 'methods_to_papers.csv'
meths_to_pprs = []
tasks_to_pprs_fn = 'tasks_to_papers.csv'
tasks_to_pprs = []
meths_orig_fn = 'methods.json'
meth_name_to_url = dict()
meth_full_name_to_url = dict()

# build method lookup table
with open(os.path.join(in_dir, meths_orig_fn)) as f:
    meths_orig = json.load(f)
dups = [set(), set()]
for meth in meths_orig:
    if meth['full_name'] in meth_full_name_to_url:
        dups[0].add(meth['full_name'])
    if meth['name'] in meth_name_to_url:
        dups[1].add(meth['name'])
    meth_name_to_url[meth['name']] = meth['url']
    meth_full_name_to_url[meth['full_name']] = meth['url']
print(f'{len(dups[0])} duplicate full method names:')
print(', '.join(dups[0]))
print(f'{len(dups[1])} duplicate method names:')
print(', '.join(dups[1]))

with open(os.path.join(in_dir, pprs_orig_fn)) as f:
    pprs_orig = json.load(f)

invalid_meth_refs = set()
for ppr in pprs_orig:
    # create preprocessed dataset object
    ppr_new = {
        key: val
        for key, val in ppr.items()
        if key not in ['tasks', 'methods']  # remove
    }
    # add URL slug ID
    ppr_id = url_to_pwc_id(ppr['paper_url'])
    ppr_new['id'] = ppr_id
    # build new dataset list
    pprs_new.append(ppr_new)
    # build methods->papers
    for meth in ppr['methods']:
        if meth['full_name'] in meth_full_name_to_url:
            meth_url = meth_full_name_to_url[meth['full_name']]
        elif meth['name'] in meth_name_to_url:
            meth_url = meth_name_to_url[meth['name']]
        else:
            invalid_meth_refs.add(meth['name'])
        meth_id = url_to_pwc_id(meth_url)
        meths_to_pprs.append(
            [meth_id, ppr_id]
        )
    # build taks->papers
    # ... here we'll need stuff from eval tbls

print(f'{len(invalid_meth_refs)} invalid method references:')
print(', '.join(invalid_meth_refs))

with open(os.path.join(out_dir, pprs_new_fn), 'w') as f:
    for ppr in pprs_new:
        json.dump(ppr, f)
        f.write('\n')

with open(os.path.join(out_dir, meths_to_pprs_fn), 'w') as f:
    csv_writer = csv.writer(
        f,
        delimiter=',',
        quoting=csv.QUOTE_NONE)
    csv_writer.writerow([
        'method_id',
        'paper_id'
    ])
    for (meth_id, ppr_id) in meths_to_pprs:
        csv_writer.writerow([meth_id, ppr_id])
