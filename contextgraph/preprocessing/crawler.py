import os
import json
import random
import re
import requests
import time
from contextgraph import config as cg_config


def _get_dataset_id(url, dataset_id_patt):
    ret = requests.get(url)
    m = dataset_id_patt.search(ret.text)
    if not m:
        return False
    return m.group(1)


def _get_dataset_papers(api_base_url, did):
    paper_dict_list = []
    start_url = f'{api_base_url}{did}'
    # first results page
    ret = requests.get(start_url).json()
    ppr_count = ret.get('count', 0)
    print(f'\tretrieving data of {ppr_count} papers')
    paper_dict_list.extend(ret.get('results', []))
    next_url = ret.get('next', None)
    # further result pages
    while next_url is not None:
        ret = requests.get(next_url).json()
        paper_dict_list.extend(ret.get('results', []))
        next_url = ret.get('next', None)
        print(f'\t{ppr_count - len(paper_dict_list)} to go')
        time.sleep(random.randint(3, 11)/10)
    return paper_dict_list


def crawl_dataset_papers():
    checkpoint_fn = 'datasets_ext_lastcheckpoint.json'
    checkpoint_interval = 100
    dataset_id_patt = re.compile(
        r'^\s*const\s*DATATABLE_PAPERS_FILTER_VALUE\s*=\s*\'(\d+)\';\s*$',
        re.M
    )
    api_base_url = cg_config.pwc_api_base_url

    with open(os.path.join(
        cg_config.pwc_data_dir,
        cg_config.pwc_dsets_fn
    )) as f:
        datasets = json.load(f)

    # incremental crawling if start checkpoint is given
    start_fp = os.path.join(cg_config.pwc_data_dir, checkpoint_fn)
    if os.path.isfile(start_fp):
        print(f'starting from checkpoint "{start_fp}"')
        with open(start_fp) as f:
            datasets_ext = json.load(f)
    else:
        print('starting without checkpoint')
        datasets_ext = dict()

    # go through all data sets
    for i, dataset in enumerate(datasets):
        print('-=[{}]=-'.format(dataset['name']))
        dataset_url = dataset['url']
        if dataset_url in datasets_ext:
            print('already done. skipping ...')
            continue
        # retrieve ID for API query
        dataset_id = _get_dataset_id(dataset_url, dataset_id_patt)
        if not dataset_id:
            continue
        datasets_ext[dataset_url] = dataset.copy()
        # extend with using papers
        datasets_ext[dataset_url]['using_papers'] = _get_dataset_papers(
            api_base_url,
            dataset_id
        )
        # behave
        time.sleep(random.randint(7, 27)/10)
        # save checkpoint
        if checkpoint_interval > 0 and i % checkpoint_interval == 0:
            print('saving checkpoint')
            with open(
                os.path.join(cg_config.pwc_data_dir, checkpoint_fn),
                'w'
            ) as f:
                json.dump(datasets_ext, f)

    # convert from dict to list
    dataset_ext_list = list(datasets_ext.values())
    with open(
        os.path.join(cg_config.pwc_data_dir, cg_config.pwc_dsets_ext_fn),
        'w'
    ) as f:
        json.dump(dataset_ext_list, f)

    print('done')
