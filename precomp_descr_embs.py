""" Script for pre computation of node description embeddings
"""

import os
import json
import networkx as nx
import numpy as np
from hashlib import md5
from sentence_transformers import SentenceTransformer
from contextgraph.util.graph import _load_node_tuples
from contextgraph.util.torch import _get_artifact_description


node_tuples = _load_node_tuples(entities_only=True)

G_lookup = nx.Graph()
G_lookup.add_nodes_from(node_tuples)
# get node descriptions
node_descrs = [
    _get_artifact_description(ntup[1], G_lookup)
    for ntup in node_tuples
]

model = SentenceTransformer('all-mpnet-base-v2')
nodeid_to_fn = {}
base_dir = 'combigraph_mpnet-base_embeddings'
for i, ntup in enumerate(node_tuples):
    node_id = ntup[0]
    node_descr = node_descrs[i]
    fn = md5(node_id.encode('utf-8')).hexdigest()
    fn += '.npy'
    nodeid_to_fn[node_id] = fn
    fp = os.path.join(base_dir, fn)
    # get embedding
    embedding = model.encode(node_descr)
    # save to disk
    np.save(fp, embedding)
    # if i % 100 == 0:
    #     print(i)

# save node ID to fn map to disk
with open(os.path.join(base_dir, 'nodeid_to_fn.json'), 'w') as f:
    json.dump(nodeid_to_fn, f)
