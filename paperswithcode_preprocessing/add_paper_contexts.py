""" Match entities from Papers With Code in unarXive paper plaintexts
"""

import argparse
import csv
import os
import json
import re
import regex
from collections import OrderedDict
from functools import lru_cache

# import sys
# sys.exit()  # WIP


def load_pwc_arxiv_papers(pwc_dir):
    pwc_pprs_fn = 'papers.jsonl'
    with open(os.path.join(pwc_dir, pwc_pprs_fn)) as f:
        pprs = [json.loads(l) for l in f]
        arxiv_pprs = [p for p in pprs if p['arxiv_id'] is not None]
    return arxiv_pprs


def load_pwc_entities(pwc_dir):
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
                    key=lambda e: len(e[1]['name'])
                )
            )
            entity_dicts.append(entity_dict)
    return entity_dicts


def load_pwc_entity_links(pwc_dir):
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


def match(pwc_dir, unarXive_dir):
    """ Match entities from Papers With Code in unarXive paper plaintexts.
    """

    # load papers to search in
    pwc_arxiv_pprs = load_pwc_arxiv_papers(pwc_dir)
    # load entities
    meths, dsets, tasks, modls = load_pwc_entities(pwc_dir)
    # load links
    pprs_to_meths, \
        pprs_to_dsets, \
        pprs_to_tasks, \
        pprs_to_modls = load_pwc_entity_links(pwc_dir)
    # output
    contexts_fn = 'contexts.jsonl'

    # write to contexts.jsonl
    # {
    #    'paper_arxiv_id': <ppr_aid>,
    #    'paper_pwc_id': <ppr_pid>,
    #    'entity_id': <entity_id>,
    #    'entity_offset': [<from>,<to>],
    #    'context_offset': [<from>,<to>],
    #    'context': <context_of_some_length>
    # }

    print(f'{len(pwc_arxiv_pprs):,} papers to get contexts from')
    print(f'{len(meths):,} unique methods')
    print(f'{len(dsets):,} unique datasets')
    print(f'{len(tasks):,} unique tasks')
    print(f'{len(modls):,} unique models')
    print(f'method-paper links for {len(pprs_to_meths):,} papers')
    print(f'dataset-paper links {len(pprs_to_dsets):,} papers')
    print(f'tasks-paper links for {len(pprs_to_tasks):,} papers')
    print(f'model-paper links for {len(pprs_to_modls):,} papers')

    import sys
    sys.exit()  # WIP

    @lru_cache(maxsize=None)
    def get_compiled_regext_patt(entity_name, flags):
        return regex.compile(
            (
             r'(?<!>)'   # prevent re-tagging of (*_used) as (*_named)
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

    for i, ppr in enumerate(pwc_arxiv_pprs):
        if i % 10000 == 0:
            print(i)

        # get entity names to match
        methods = ppr['methods']
        datasets = ppr['datasets']
        method_names = sorted(
            [m['name'] for m in methods],
            key=len,
            reverse=True
        )
        dataset_names = sorted(
            [d['name'] for d in datasets],
            key=len,
            reverse=True
        )
        task_names = sorted(ppr['tasks'], key=len, reverse=True)

        # get plaintext
        paper_fn_id = ppr['arxiv_id'].replace('/', '')
        paper_fn = f'{paper_fn_id}.txt'
        paper_path = os.path.join(unarXive_dir, paper_fn)
        if not os.path.isfile(paper_path):
            continue
        with open(paper_path) as f:
            paper_text = f.read()

        annotation_types = {
            'method_used': method_names,
            'dataset_used': dataset_names,
            'task_used': task_names,
            'method_named': all_method_names,
            'dataset_named': all_dataset_names,
            'task_named': all_task_names
        }
        debug_used_entities = {
            'method_names': method_names,
            'dataset_names': dataset_names,
            'task_names': task_names
        }
        subs_per_entity = dict()
        for tag_name, entity_names in annotation_types.items():
            for entity_name in entity_names:
                # skip the few two character entities that do not
                # include numbers (i.e. not T5)
                if len(entity_name) < 3 and not re.search(r'\d', entity_name):
                    continue
                # try to match as much as possible case insensitive
                if re.search(r'\d', entity_name):
                    regex_flags = re.I  # insensitive if there's a number in it
                elif (
                    # match case sensitive if all upper case (NICE, SECOND)
                    entity_name.upper() == entity_name or
                    # entity names that resemble common words (ZeRO, ReCoRD)
                    entity_name in naughty_entity_names
                ):
                    regex_flags = 0
                else:
                    regex_flags = re.I
                patt = get_compiled_regext_patt(entity_name, regex_flags)
                paper_text, num_subs = patt.subn(
                    r'<{0}>\1</{0}>'.format(tag_name),
                    paper_text
                )
                if '_used' in tag_name or num_subs > 0:
                    # only write debug info for entities to expect
                    # or those additionally found (mentioned but not used)
                    subs_per_entity[tag_name + '///' + entity_name] = num_subs

        paper_out_fn = f'{paper_fn_id}.txt'
        paper_out_path = os.path.join(out_dir, paper_out_fn)
        with open(paper_out_path, 'w') as f:
            f.write(paper_text)

        # write additional triple information
        triples, debug_info = build_triples(ppr)
        triple_fn = f'{paper_fn_id}_rels.tsv'
        triple_path = os.path.join(out_dir, triple_fn)
        with open(triple_path, 'w') as f:
            for trpl in triples:
                f.write('\t'.join(trpl) + '\n')
        debug_info_fn = f'{paper_fn_id}_debug.json'
        debug_info_path = os.path.join(out_dir, debug_info_fn)
        debug_info['subs_per_entity'] = subs_per_entity
        debug_info['used_entities_pwc'] = debug_used_entities
        with open(debug_info_path, 'w') as f:
            json.dump(debug_info, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('script that extracts entity mention contexts'
                     'unarXive paper plain texts')
    )
    parser.add_argument(
        '--pwc_dir',
        help=('path to the directory with the preprocessed Papers With Code '
              'JSON files'),
        required=True
    )
    parser.add_argument(
        '--unarxive_dir',
        help='path to the directory with the unarXive plain text files',
        required=True
    )
    args = parser.parse_args()
    match(args.pwc_dir, args.unarxive_dir)
