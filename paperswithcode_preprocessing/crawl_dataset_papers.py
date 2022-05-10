import os
import json
import random
import re
import requests
import time


base_path = '/home/ws/ys8950/dev/data/paperswithcode/data/'
datasets_fn = 'datasets.json'
datasets_ext_fn = 'datasets_ext.json'
checkpoint_fn = 'datasets_ext_lastcheckpoint.json'
checkpoint_interval = 100
dataset_id_patt = re.compile(
    r'^\s*const\s*DATATABLE_PAPERS_FILTER_VALUE\s*=\s*\'(\d+)\';\s*$',
    re.M
)
api_base_url = ('https://paperswithcode.com/api/internal/papers/'
                '?format=json&paperdataset__dataset_id=')


def get_dataset_id(url):
    ret = requests.get(url)
    m = dataset_id_patt.search(ret.text)
    if not m:
        return False
    return m.group(1)


def get_dataset_papers(did):
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


with open(os.path.join(base_path, datasets_fn)) as f:
    datasets = json.load(f)

# incremental crawling if start checkpoint is given
start_fp = os.path.join(base_path, checkpoint_fn)
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
    dataset_id = get_dataset_id(dataset_url)
    if not dataset_id:
        continue
    datasets_ext[dataset_url] = dataset.copy()
    # extend with using papers
    datasets_ext[dataset_url]['using_papers'] = get_dataset_papers(dataset_id)
    # behave
    time.sleep(random.randint(7, 27)/10)
    # save checkpoint
    if checkpoint_interval > 0 and i % checkpoint_interval == 0:
        print('saving checkpoint')
        with open(os.path.join(base_path, checkpoint_fn), 'w') as f:
            json.dump(datasets_ext, f)


with open(os.path.join(base_path, datasets_ext_fn), 'w') as f:
    json.dump(datasets_ext, f)

print('done')
