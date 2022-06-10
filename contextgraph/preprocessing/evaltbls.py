""" From
        evaluation-tables.json
    generate
        tasks.jsonl  (removes tasks_pre.jsonl)
        models.jsonl
        methods_to_datasets.csv
        models_to_papers_pre.csv
"""

import csv
import json
import os
import re
from contextgraph import config as cg_config
from contextgraph.util.preprocessing import name_to_slug


def preprocess_evaltables():
    in_dir = cg_config.pwc_data_dir
    out_dir = cg_config.graph_data_dir

    evals_orig_fn = cg_config.pwc_evals_fn
    meths_processed_fn = cg_config.graph_meths_fn
    meth_name_to_id = dict()
    dsets_processed_fn = cg_config.graph_dsets_fn
    dset_name_to_id = dict()
    tasks_preprocessed_fn = cg_config.graph_tasks_pre_fn
    task_name_to_id = dict()
    tasks_new_fn = cg_config.graph_tasks_fn
    tasks_new = []
    tasks_to_subtasks_fn = cg_config.graph_tasks_to_subtasks_fn
    modls_new_fn = cg_config.graph_modls_fn
    modls_new = dict()
    meths_to_dsets_fn = cg_config.graph_meths_to_dsets_fn
    meths_to_dsets = []
    modls_to_pprs_fn = cg_config.graph_modls_to_pprs_pre_fn
    modls_to_pprs = []

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
        task_name_to_id = {
            task['name']: task['id']
            for task in [json.loads(line) for line in lines]
        }

    with open(os.path.join(in_dir, evals_orig_fn)) as f:
        evals = json.load(f)

    def recursively_process_eval_list(evals):
        # start empty (returns empty at deepest level
        # because evals will be an empty list)
        tasks = []
        tasks_to_subtasks = []
        for evl in evals:
            # add accumulated info from deeper levels
            sub_ts, sub_ts_to_subts = recursively_process_eval_list(
                evl['subtasks']
            )
            tasks += sub_ts
            tasks_to_subtasks += sub_ts_to_subts
            # add task to subtask link from this level
            for subtask_name in [subevl['task'] for subevl in evl['subtasks']]:
                tasks_to_subtasks.append(
                    [evl['task'], subtask_name]
                )
            # add task from this level
            task = {
                'name': evl['task'],
                'type': 'task',
                'description': evl['description'],
                'categories': evl['categories'],
                'dsets_tmp': []
            }
            for dset in evl['datasets']:
                dset_tmp = {
                    'name': dset['dataset'],
                    'type': 'dataset',
                    'lnks_tmp': [l['url'] for l in dset['dataset_links']],
                    'mdls_tmp': []
                }
                for sota_row in dset['sota']['rows']:
                    mdl_tmp = {
                        'name': sota_row['model_name'],
                        'type': 'model',
                        'paper_date': sota_row['paper_date'],
                        'paper_title': sota_row['paper_title'],
                        'paper_url': sota_row['paper_url']
                    }
                    dset_tmp['mdls_tmp'].append(mdl_tmp)
                task['dsets_tmp'].append(dset_tmp)
            tasks.append(task)
        return tasks, tasks_to_subtasks

    eval_tasks, tasks_to_subtasks = recursively_process_eval_list(evals)

    for eval_task in eval_tasks:
        # create task entity
        task_id = task_name_to_id.get(eval_task['name'], None)
        if task_id is None:
            new_slug = name_to_slug(eval_task['name'])
            task_id = 'pwc:task/' + new_slug
            # add to map for later conversion of task to subtask links
            task_name_to_id[eval_task['name']] = task_id

        task_new = {
            'id': task_id,
            'type': 'task',
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
                # if this model is also a method, we create some special
                # links, because we then know that
                # - a method was evaluated
                # - on a certain data set
                # - (if there is a paper given:) at a certain point in time
                if eval_modl['name'] in meth_name_to_id:
                    # model is also treated as a method by PWC
                    # -> create method to dataset link

                    #    (if not a sub dset
                    #     TODO: can we identify the parent dset?
                    #           there is a subdataset dict key but
                    #           never used in evaluation-tables.json)
                    meth_id = meth_name_to_id[eval_modl['name']]
                    # TODO: check for paper_date None or not and
                    #       only create link if there is one?
                    # consideration to make: do we want to perform link
                    # prediction for predicting
                    # a) "combined" use in a more general sense
                    # or
                    # b) combined use (in an ML eval)
                    if not is_sub_dset:
                        # it’s a method
                        if eval_modl['paper_date'] is not None:
                            # and we have a data to associate with it
                            eval_date = eval_modl['paper_date']
                            eval_y = int(eval_date[:4])
                            eval_m = int(eval_date[5:7])
                            eval_d = int(eval_date[8:])
                            csv_safe_title = re.sub(
                                r'\W',
                                '_',
                                eval_modl['paper_title']
                            )
                            meths_to_dsets.append([
                                meth_id,
                                dset_id,
                                csv_safe_title,
                                eval_date,
                                eval_y,
                                eval_m,
                                eval_d,
                                task_id
                            ])
                            # ^ cries for a eval node, but papers are
                            # probably not easily matchable by URL
                            # or title
                modl_id = 'pwc:model/' + name_to_slug(eval_modl['name'])
                modl = modls_new.get(modl_id, None)
                if modl is None:
                    # first time we see this. create new entity
                    modl = {
                        'id': modl_id,
                        'type': 'model',
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
                # also create a link from model to paper
                modls_to_pprs.append([
                    modl_id,
                    eval_modl['paper_url']  # matches to "url_abs"
                ])                          # in ppr entities

    # replace names in task to subtask links with task IDs
    tasks_to_subtasks_id = []
    for link in tasks_to_subtasks:
        tasks_to_subtasks_id.append([
            task_name_to_id[link[0]],
            task_name_to_id[link[1]]
        ])
    tasks_to_subtasks = tasks_to_subtasks_id

    with open(os.path.join(out_dir, tasks_new_fn), 'w') as f:
        for task in tasks_new:
            json.dump(task, f)
            f.write('\n')

    with open(os.path.join(out_dir, modls_new_fn), 'w') as f:
        for modl_id, modl in modls_new.items():
            modl['using_paper_titles'] = list(modl['using_paper_titles'])
            json.dump(modl, f)
            f.write('\n')

    with open(os.path.join(out_dir, tasks_to_subtasks_fn), 'w') as f:
        csv_writer = csv.writer(
            f,
            delimiter=',',
            quoting=csv.QUOTE_NONE
        )
        csv_writer.writerow([
            'task_id',
            'subtask_id',
        ])
        for (task_id, subtask_id) in tasks_to_subtasks:
            csv_writer.writerow([task_id, subtask_id])

    with open(os.path.join(out_dir, meths_to_dsets_fn), 'w') as f:
        csv_writer = csv.writer(
            f,
            delimiter=',',
            quoting=csv.QUOTE_NONE
        )
        csv_writer.writerow([
            'method_id',
            'dataset_id',
            'eval_paper_title',
            'eval_date',
            'eval_year',
            'eval_month',
            'eval_day',
            'task_id'
        ])
        for vals in meths_to_dsets:
            csv_writer.writerow(vals)

    with open(os.path.join(out_dir, modls_to_pprs_fn), 'w') as f:
        csv_writer = csv.writer(
            f,
            delimiter=',',
            quoting=csv.QUOTE_NONE
        )
        csv_writer.writerow([
            'model_id',
            'paper_url',
        ])
        for (modl_id, ppr_url) in modls_to_pprs:
            csv_writer.writerow([modl_id, ppr_url])

    os.remove(os.path.join(out_dir, tasks_preprocessed_fn))
