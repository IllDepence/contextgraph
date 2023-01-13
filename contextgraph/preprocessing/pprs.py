""" From
        papers.json
    generate
        papers.jsonl
        tasks_to_papers.csv
        methods_to_papers.csv
        models_to_papers.csv  (removes models_to_papers_pre.csv)
    extend
        tasks.jsonl
"""

import csv
import json
import os
from contextgraph import config as cg_config
from contextgraph.util.preprocessing import url_to_pwc_id, name_to_slug


def preprocess_papers(verbose=False):
    in_dir = cg_config.pwc_data_dir
    out_dir = cg_config.graph_data_dir

    pprs_orig_fn = cg_config.pwc_pprs_fn
    pprs_orig = []
    pprs_new_fn = cg_config.graph_pprs_fn
    pprs_new = []
    meths_to_pprs_fn = cg_config.graph_meths_to_pprs_fn
    meths_to_pprs = []
    tasks_to_pprs_fn = cg_config.graph_tasks_to_pprs_fn
    tasks_to_pprs = []
    meths_orig_fn = cg_config.pwc_meths_fn
    meth_name_to_url = dict()
    meth_full_name_to_url = dict()
    tasks_preprocessed_fn = cg_config.graph_tasks_fn
    task_name_to_id = dict()
    modls_to_pprs_pre_fn = cg_config.graph_modls_to_pprs_pre_fn
    modls_to_pprs_pre = []
    modls_to_pprs_fn = cg_config.graph_modls_to_pprs_fn
    modls_to_pprs = []
    ppr_abs_url_to_id = dict()

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

    if verbose:
        print('- - - - - method reference data - - - - -')
        print(f'{len(dups[0])} duplicate full method names:')
        print(', '.join(dups[0]))
        print(f'{len(dups[1])} duplicate method names:')
        print(', '.join(dups[1]))

    # build task lookup dict
    with open(os.path.join(out_dir, tasks_preprocessed_fn)) as f:
        tasks = [json.loads(line) for line in f]
        task_name_to_id = {
            task['name']: task['id']
            for task in tasks
        }

    # get prerpocessed model to paper links
    with open(os.path.join(out_dir, modls_to_pprs_pre_fn)) as f:
        csv_reader = csv.DictReader(
            f,
            delimiter=',',
            quoting=csv.QUOTE_ALL  # written with QUOTE_ALL because paper URLs
            #                        may need escaping. everywhere else we use
            #                        QUOTE_NONE b/c PwC IDs are safe
        )
        for row in csv_reader:
            modls_to_pprs_pre.append([
                row['model_id'],
                row['paper_url']
            ])

    with open(os.path.join(in_dir, pprs_orig_fn)) as f:
        pprs_orig = json.load(f)

    invalid_meth_refs = set()
    known_task_refs = set()
    tasks_new = dict()
    id_shiftet_tasks = set()
    for ppr in pprs_orig:
        # create paper object
        ppr_new = {
            key: val
            for key, val in ppr.items()
            if key not in ['tasks', 'methods']  # remove
        }
        # add URL slug ID and other attributes
        ppr_id = url_to_pwc_id(ppr['paper_url'])
        ppr_new['id'] = ppr_id
        ppr_new['type'] = 'paper'
        if len(ppr['date']) > 0:
            ppr_new['year'] = int(ppr['date'][:4])
            ppr_new['month'] = int(ppr['date'][5:7])
            ppr_new['day'] = int(ppr['date'][8:])
        else:
            ppr_new['year'] = -1
            ppr_new['month'] = -1
            ppr_new['day'] = -1
        # build paper list
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
        # + extend tasks
        for task_name in ppr['tasks']:
            if task_name in task_name_to_id:
                task_id = task_name_to_id[task_name]
                known_task_refs.add(task_name)
            else:
                task_id = 'pwc:task/' + name_to_slug(task_name)
                try:
                    assert(task_id not in task_name_to_id.values())
                except AssertionError:
                    # problem example:
                    # Source Code Summarization slug: task/code-summarization
                    # Code Summarization slug:        task/code-summarization-1
                    # -.-
                    id_counter = 1
                    id_appendage = f'-{id_counter}'
                    while task_id in task_name_to_id.values():
                        task_id = task_id + id_appendage
                        id_counter += 1
                        id_appendage = f'-{id_counter}'
                    id_shiftet_tasks.add(task_name)
                # new task entity
                tasks_new[task_id] = {
                    'id': task_id,
                    'type': 'task',
                    'name': task_name,
                    'description': None,
                    'categories': []
                }
            tasks_to_pprs.append([
                task_id,
                ppr_id
            ])
        # prepare mapping to finalize model to paper links
        ppr_abs_url_to_id[ppr['url_abs']] = ppr_id

    # create final model to paper links
    num_non_linkable_pprs = 0
    for (modl_id, ppr_abs_url) in modls_to_pprs_pre:
        ppr_id = ppr_abs_url_to_id.get(ppr_abs_url, None)
        if ppr_id is not None:
            modls_to_pprs.append([
                modl_id,
                ppr_id
            ])
        else:
            num_non_linkable_pprs += 1

    if verbose:
        print('- - - - - model reference data - - - - -')
        print((f'could not convert {num_non_linkable_pprs:,} '
               f'of {len(modls_to_pprs_pre):,} model to paper links'))

    # add new unique tasks to task list
    tasks.extend(tasks_new.values())

    if verbose:
        print('- - - - - method association - - - - -')
        print(f'{len(invalid_meth_refs)} invalid method references:')
        print(', '.join(invalid_meth_refs))

        print('- - - - - task association - - - - -')
        print(f'{len(known_task_refs)} known task references')
        print(f'{len(tasks_new)} newly created tasks')
        print(f'{len(id_shiftet_tasks)} task with shifted ID:')
        print(', '.join(id_shiftet_tasks))

    with open(os.path.join(out_dir, pprs_new_fn), 'w') as f:
        for ppr in pprs_new:
            json.dump(ppr, f)
            f.write('\n')

    with open(os.path.join(out_dir, tasks_preprocessed_fn), 'w') as f:
        for task in tasks:
            json.dump(task, f)
            f.write('\n')

    with open(os.path.join(out_dir, meths_to_pprs_fn), 'w') as f:
        csv_writer = csv.writer(
            f,
            delimiter=',',
            quoting=csv.QUOTE_NONE
        )
        csv_writer.writerow([
            'method_id',
            'paper_id'
        ])
        for (meth_id, ppr_id) in meths_to_pprs:
            csv_writer.writerow([meth_id, ppr_id])

    with open(os.path.join(out_dir, tasks_to_pprs_fn), 'w') as f:
        csv_writer = csv.writer(
            f,
            delimiter=',',
            quoting=csv.QUOTE_NONE
        )
        csv_writer.writerow([
            'task_id',
            'paper_id'
        ])
        for (task_id, ppr_id) in tasks_to_pprs:
            csv_writer.writerow([task_id, ppr_id])

    with open(os.path.join(out_dir, modls_to_pprs_fn), 'w') as f:
        csv_writer = csv.writer(
            f,
            delimiter=',',
            quoting=csv.QUOTE_NONE
        )
        csv_writer.writerow([
            'model_id',
            'paper_id'
        ])
        for (modl_id, ppr_id) in modls_to_pprs:
            csv_writer.writerow([modl_id, ppr_id])

    os.remove(os.path.join(out_dir, modls_to_pprs_pre_fn))
