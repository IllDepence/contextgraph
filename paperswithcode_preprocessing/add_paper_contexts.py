""" Match entities from Papers With Code in unarXive paper plaintexts
"""

import argparse
import os
import json
import re
import regex
import pandas as pd
from functools import lru_cache

import sys
sys.exit()  # WIP


def match(pwc_dir, unarXive_dir):
    """ Match entities from Papers With Code in unarXive paper plaintexts.
    """

    pwc_meths_fn = 'methods.jsonl'
    pwc_dsets_fn = 'datasets.jsonl'
    pwc_tasks_fn = 'tasks.jsonl'
    pwc_modls_fn = 'models.jsonl'

    with open(os.path.join(pwc_dir, pwc_meths_fn)) as f:
        meths = [json.loads(l) for l in f]
    # for each entity type:
    # sort entity list by length of entity name,
    # go through all papers
    # write to contexts.jsonl
    # {
    #    'paper_arxiv_id': <ppr_aid>,
    #    'paper_pwc_id': <ppr_pid>,
    #    'entity_id': <entity_id>,
    #    'entity_offset': [<from>,<to>],
    #    'context_offset': [<from>,<to>],
    #    'context': <context_of_some_length>
    # }


    # --- old code ---
    all_method_names = sorted(all_method_names, key=len, reverse=True)
    all_dataset_names = df_paper_abstracts_extended['datasets'].apply(
        lambda x: [d['name'] for d in x]
    ).explode().dropna().unique()
    all_dataset_names = sorted(all_dataset_names, key=len, reverse=True)
    all_task_names = df_paper_abstracts_extended[
        'tasks'
    ].explode().dropna().unique()
    all_task_names = sorted(all_task_names, key=len, reverse=True)

    print(f'{len(all_method_names):,} unique methods')
    print(f'{len(all_dataset_names):,} datasets methods')
    print(f'{len(all_task_names):,} tasks methods')

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

    for i, ppr in df_paper_abstracts_extended[
        df_paper_abstracts_extended.arxiv_id.notna()
    ].iterrows():
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
