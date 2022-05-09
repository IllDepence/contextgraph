import json
import os
from util import url_to_slug

base_dir = '/home/ls3data/datasets/paperswithcode/'

meths_orig_fn = 'methods.json'
meths_orig = []
meths_new_fn = 'methods.jsonl'
meths_new = []
meths_to_colls_fn = 'methods_to_collections.jsonl'
meths_to_colls = {}
meth_areas_fn = 'method_areas.jsonl'
meth_areas = {}

with open(os.path.join(base_dir, meths_orig_fn)) as f:
    meths_orig = json.load(f)

for meth in meths_orig:
    meth_new = {
        key: meth[key]
        for key in meth
        if key not in ['collections']  # remove collections
    }
    # add URL slug ID
    meth_new['id'] = url_to_slug(meth['url'])
    # build area->collection
    for coll in meth['collections']:
        coll_id = coll['collection'].lower().replace(' ', '-')
        if coll['area-id'] not in meth_areas:
            meth_areas['area-id'] = {
                'id': meth_areas['area-id'],
                'name': coll['area'],
                'collections': []
            }
        meth_areas['area-id']['collections'].append(
            {'id': coll_id, 'name': coll['collection']}
        )

# TODO: persist result, test
