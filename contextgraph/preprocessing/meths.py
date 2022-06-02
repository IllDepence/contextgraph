""" From
        methods.json
    generate
        methods.jsonl
        methods_to_collections.csv
        method_areas.jsonl
"""

import csv
import json
import os
from contextgraph import config as cg_config
from contextgraph.util.preprocessing import url_to_pwc_id


def preprocess_methods():
    in_dir = cg_config.pwc_data_dir
    out_dir = cg_config.graph_data_dir

    meths_orig_fn = cg_config.pwc_meths_fn
    meths_orig = []
    meths_new_fn = cg_config.graph_meths_fn
    meths_new = []
    meths_to_colls_fn = cg_config.graph_meths_to_colls_fn
    meths_to_colls = {}
    meth_areas_fn = cg_config.graph_meth_areas_fn
    meth_areas = {}

    with open(os.path.join(in_dir, meths_orig_fn)) as f:
        meths_orig = json.load(f)

    for meth in meths_orig:
        # create preprocessed method object
        meth_new = {
            key: val
            for key, val in meth.items()
            if key not in ['collections']  # remove collections
        }
        # add URL slug ID
        meth_id = url_to_pwc_id(meth['url'])
        meth_new['id'] = meth_id
        meth_new['type'] = 'method'
        # build new methods list
        meths_new.append(meth_new)
        # build method->collection
        #   and area->collection
        area_prefix = 'pwc:area/'
        coll_prefix = 'pwc:collection/'
        meths_to_colls[meth_id] = []
        for coll in meth['collections']:
            coll_id = coll_prefix \
                + coll['collection'].lower().replace(' ', '-')
            area_id = area_prefix + coll['area_id']
            meths_to_colls[meth_id].append(coll_id)
            meths_to_colls
            if area_id not in meth_areas:
                meth_areas[area_id] = {
                    'id': area_id,
                    'type': 'area',
                    'name': coll['area'],
                    'collections': []
                }
            # insert w/o checks here, remove duplicates later
            meth_areas[area_id]['collections'].append(
                {
                    'id': coll_id,
                    'type': 'connection',
                    'name': coll['collection']
                }
            )

    with open(os.path.join(out_dir, meths_new_fn), 'w') as f:
        for meth in meths_new:
            json.dump(meth, f)
            f.write('\n')

    with open(os.path.join(out_dir, meths_to_colls_fn), 'w') as f:
        csv_writer = csv.writer(
            f,
            delimiter=',',
            quoting=csv.QUOTE_NONE
        )
        csv_writer.writerow([
            'method_id',
            'collection_id',
        ])
        for meth, colls in meths_to_colls.items():
            for coll in colls:
                csv_writer.writerow([meth, coll])

    with open(os.path.join(out_dir, meth_areas_fn), 'w') as f:
        for area_id, area in meth_areas.items():
            # remove duplicates
            area['collections'] = [
                dict(tup) for tup in {
                    tuple(coll.items()) for coll in area['collections']
                }
            ]
            json.dump(area, f)
            f.write('\n')
