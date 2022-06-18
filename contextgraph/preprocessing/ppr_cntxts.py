""" From
        papers.jsonl
    and
        the unarXive plain text paper TXTs
    generate
        contexts_used.jsonl
        contexts_mentioned.jsonl  (experimental)
"""

import csv
import os
import json
import re
import regex
from collections import OrderedDict
from functools import lru_cache
import contextgraph.config as cg_config
import numpy as np
# import nltk
# from nltk.tokenize import sent_tokenize
# nltk.download('punkt')

# LENGTH_OF_LETTERS = 1000


def _load_pwc_arxiv_papers(pwc_dir):
    pwc_pprs_fn = 'papers.jsonl'
    with open(os.path.join(pwc_dir, pwc_pprs_fn)) as f:
        pprs = [json.loads(l) for l in f]
        arxiv_pprs = [p for p in pprs if p['arxiv_id'] is not None]
    return arxiv_pprs


def _load_pwc_entities(pwc_dir):
    pwc_meths_fn = 'methods.jsonl'
    pwc_dsets_fn = 'datasets.jsonl'
    pwc_tasks_fn = 'tasks.jsonl'
    pwc_modls_fn = 'models.jsonl'
    entity_dicts = []
    for fn in [pwc_meths_fn, pwc_dsets_fn, pwc_tasks_fn, pwc_modls_fn]:
        with open(os.path.join(pwc_dir, fn)) as f:
            entity_dict_unsorted = OrderedDict()
            for line in f:
                entity = json.loads(line)
                entity_dict_unsorted[entity['id']] = entity
            # sort by length to prioritize specific terms
            entity_dict = OrderedDict(
                sorted(
                    entity_dict_unsorted.items(),
                    key=lambda e: len(e[1]['name']),
                    reverse=True
                )
            )
            entity_dicts.append(entity_dict)
    return entity_dicts


def _load_pwc_entity_links(pwc_dir):
    meths_to_pprs_fn = 'methods_to_papers.csv'
    dsets_to_pprs_fn = 'datasets_to_papers.csv'
    tasks_to_pprs_fn = 'tasks_to_papers.csv'
    modls_to_pprs_fn = 'models_to_papers.csv'
    links = []
    for fn in [meths_to_pprs_fn, dsets_to_pprs_fn,
               tasks_to_pprs_fn, modls_to_pprs_fn]:
        with open(os.path.join(pwc_dir, fn)) as f:
            csv_reader = csv.DictReader(
                f,
                delimiter=',',
                quoting=csv.QUOTE_NONE
            )
            headers = csv_reader.fieldnames
            links_single = dict()
            for row in csv_reader:
                entity_id = row[headers[0]]
                ppr_id = row[headers[1]]
                if ppr_id not in links_single:
                    links_single[ppr_id] = []
                links_single[ppr_id].append(entity_id)
            links.append(links_single)
    return links


def _get_context_by_sent(passage, entity_name, num_pre=1, num_suc=1):
    sentences = sent_tokenize(passage)
    positions = [i for i, s in enumerate(sentences) if entity_name in s]
    if len(positions) == 1:
        pos_entity = positions[0]
    else:
        length_ratio = np.cumsum([len(s) for i, s in enumerate(sentences)]) / (LENGTH_OF_LETTERS * 2 + len(entity_name))
        pos_in_list = np.argmin(length_ratio[positions] - 0.5)
        pos_entity = positions[pos_in_list]
    context = sentences[pos_entity-num_pre: pos_entity+num_suc+1]
    return context


def _generate_context_id(ppr_id, e_name, cntxt_start, cntxt_end):
    return 'uxv:context/{}-{}-{}-{}'.format(
        ppr_id,
        e_name,
        cntxt_start,
        cntxt_end
    )


def add_paper_contexts(verbose=False, mentioned_contexts=False):
    """ Match entities from Papers With Code in unarXive paper plaintexts.
    """

    graph_data_dir = cg_config.graph_data_dir
    unarXive_paper_dir = cg_config.unarxive_paper_dir

    # load papers to search in
    pwc_arxiv_pprs = _load_pwc_arxiv_papers(graph_data_dir)
    # load entities
    meths_dict, dsets_dict, tasks_dict, modls_dict = _load_pwc_entities(
        graph_data_dir
    )
    meths_list = meths_dict.values()
    dsets_list = dsets_dict.values()
    tasks_list = tasks_dict.values()
    modls_list = modls_dict.values()
    # load links
    pprs_to_meths, \
        pprs_to_dsets, \
        pprs_to_tasks, \
        pprs_to_modls = _load_pwc_entity_links(graph_data_dir)
    # output
    contexts_used_fn = 'contexts_used.jsonl'
    contexts_used = []
    contexts_mentioned_fn = 'contexts_mentioned.jsonl'
    contexts_mentioned = []

    if verbose:
        print(f'{len(pwc_arxiv_pprs):,} papers to get contexts from')
        print(f'{len(meths_list):,} unique methods')
        print(f'{len(dsets_list):,} unique datasets')
        print(f'{len(tasks_list):,} unique tasks')
        print(f'{len(modls_list):,} unique models')
        print(f'method-paper links for {len(pprs_to_meths):,} papers')
        print(f'dataset-paper links {len(pprs_to_dsets):,} papers')
        print(f'tasks-paper links for {len(pprs_to_tasks):,} papers')
        print(f'model-paper links for {len(pprs_to_modls):,} papers')

    @lru_cache(maxsize=None)
    def get_compiled_regext_patt(entity_name, flags):
        return regex.compile(
            (
             r'(?<=\W)'  # expect a preceding non-word character
             r'({})'     # the entity name itself
             r'(?=(\W|s\W|ed\W))'  # expect a succeeding non-word character or
            ).format(re.escape(entity_name)),  # #                 s\W or ed\W
            regex_flags
        )

    # common words but distinguishable by matching case sensitive
    naughty_entity_names = [
        'ZeRO', 'Cell', 'ReCoRD', 'Inspired', 'MaSS', 'MuTual', 'IntrA',
        'Sketch', 'Letter', 'Digits', 'VOICe', 'HoME', 'Places', 'BiRD',
        'Shifts', 'Finer', 'AND Dataset', 'Electricity', 'Atlas', 'Replica',
        'GlaS', 'eSCAPE', 'ExPose', 'Torque', 'Finer'
    ]
    # common words not even distinguishable when matching case sensitive
    super_naughty_entity_names = [
        'Google', 'seeds', 'iris', 'SSL', 'E-commerce', 'ACM'
    ]

    # go through all papers
    for i, ppr in enumerate(pwc_arxiv_pprs):
        if i % 1000 == 0 and verbose:
            print(i)

        # get entities to match
        ppr_meths = [
            meths_dict[mid] for mid in pprs_to_meths.get(ppr['id'], [])
        ]
        ppr_dsets = [
            dsets_dict[did] for did in pprs_to_dsets.get(ppr['id'], [])
        ]
        ppr_tasks = [
            tasks_dict[tid] for tid in pprs_to_tasks.get(ppr['id'], [])
        ]
        ppr_modls = [
            modls_dict[mid] for mid in pprs_to_modls.get(ppr['id'], [])
        ]

        # get plaintext
        paper_fn_id = ppr['arxiv_id'].replace('/', '')
        paper_fn = f'{paper_fn_id}.txt'
        paper_path = os.path.join(unarXive_paper_dir, paper_fn)
        if not os.path.isfile(paper_path):
            continue
        with open(paper_path) as f:
            paper_text = f.read()

        if mentioned_contexts:
            entity_types = {
                'method_used': ppr_meths,
                'dataset_used': ppr_dsets,
                'task_used': ppr_tasks,
                'model_used': ppr_modls,
                'method': meths_list,
                'dataset': dsets_list,
                'task': tasks_list,
                'model': modls_list,
            }
        else:
            entity_types = {
                'method_used': ppr_meths,
                'dataset_used': ppr_dsets,
                'task_used': ppr_tasks,
                'model_used': ppr_modls,
            }

        # go through all entities (1) of the paper and (2) in all of pwc
        for etype, entities in entity_types.items():
            for entity in entities:
                # skip the few two character entities that do not
                # include numbers (i.e. not T5)
                if len(entity['name']) < 3 \
                        and not re.search(r'\d', entity['name']):
                    continue
                # skip overly ambiguous entity names
                if entity['name'] in super_naughty_entity_names:
                    continue
                # try to match as much as possible case insensitive
                # (set regex flags accordingly)
                if re.search(r'\d', entity['name']):
                    regex_flags = re.I  # insensitive if there's a number in it
                elif (
                    # match case sensitive if all upper case (NICE, SECOND)
                    entity['name'].upper() == entity['name'] or
                    # entity names that resemble common words (ZeRO, ReCoRD)
                    entity['name'] in naughty_entity_names
                ):
                    regex_flags = 0
                else:
                    regex_flags = re.I
                patt = get_compiled_regext_patt(entity['name'], regex_flags)
                for m in patt.finditer(paper_text):
                    entity_offset_start = m.start()
                    entity_offset_end = m.end()
                    # FIXME: _get_context_by_sent throws errors
                    # context_offset_start = max(
                    #     entity_offset_start-LENGTH_OF_LETTERS,
                    #     0
                    # )
                    # context_offset_end = min(
                    #     entity_offset_end+LENGTH_OF_LETTERS,
                    #     len(paper_text)
                    # )
                    # context_passage = paper_text[
                    #     context_offset_start:context_offset_end
                    # ]
                    # context = _get_context_by_sent(context_passage, m.group(0))
                    context_offset_start = max(
                        entity_offset_start-100,
                        0
                    )
                    context_offset_end = min(
                        entity_offset_end+100,
                        len(paper_text)
                    )
                    context = paper_text[
                        context_offset_start:context_offset_end
                    ]
                    # create new context entity
                    context_entity = {
                        'id': _generate_context_id(
                            ppr['arxiv_id'],
                            entity['name'],
                            context_offset_start,
                            context_offset_end
                        ),
                        'type': 'context',
                        'paper_arxiv_id': ppr['arxiv_id'],
                        'paper_pwc_id': ppr['id'],
                        'entity_id': entity['id'],
                        'entity_offset_in_context': [
                            # FIXME: try making this shallow (separate
                            # attribs for start and end) and see if it
                            # fixes cytoscape import from JSON
                            entity_offset_start-context_offset_start,
                            entity_offset_end-context_offset_end
                         ],
                        'entity_offset_in_paper': [
                            entity_offset_start,
                            entity_offset_end
                         ],
                        'context_offset_in_paper': [
                            context_offset_start,
                            context_offset_end
                         ],
                        'context': context
                    }
                    if '_used' in etype:
                        contexts_used.append(context_entity)
                    else:
                        contexts_mentioned.append(context_entity)

    # persist contexts
    with open(os.path.join(graph_data_dir, contexts_used_fn), 'w') as f:
        for context in contexts_used:
            json.dump(context, f)
            f.write('\n')
    if mentioned_contexts:
        with open(
            os.path.join(graph_data_dir, contexts_mentioned_fn),
            'w'
        ) as f:
            for context in contexts_mentioned:
                json.dump(context, f)
                f.write('\n')
