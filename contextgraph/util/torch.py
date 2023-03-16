""" Methods for loading the graph for pytorch gemoetric
"""

import os
import json
import numpy as np
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from contextgraph.util.graph import _load_node_tuples,\
                                    _load_entity_combi_edge_tuples
from sklearn.feature_extraction.text import TfidfVectorizer


def load_full_graph():
    """ Return full graph in a form usable with torch geometric.
    """

    raise NotImplementedError


def _get_artifact_description(node_attrs, G):
    """ Return a string that is used as the description of the given
        artifact.
    """

    if node_attrs['type'] == 'model':
        # dont’t have a description in and of themselves, so we create one from
        # - the model’s name
        # - the titles of the papers that use the model
        # - the titles of the tasks the is was used for
        task_names = set()
        for (task_id, dset) in node_attrs['evaluations']:
            task = G.nodes.get(task_id)
            if task is not None:
                task_names.add(task['name'])
        tasks_insert = ' and '.join(
            task_names
        )
        pprs_insert = ' and '.join(
            node_attrs['using_paper_titles']
        )
        descr = (
            f'The model {node_attrs["name"]} '
            f'is used for the tasks {tasks_insert}. '
            f'It has been used in in the papers {pprs_insert}.'
        )
    else:
        # datasets, methods and tasks already have descriptions
        descr = node_attrs['description']

    return descr


def _embed_string_atrs_tfidf(node_attrs):
    """ Transform a list of node string attributes (one attr per node)
        into tfidf vectors
    """

    vectorizer = TfidfVectorizer(
        max_features=None,
        stop_words=None,
        # consider limiting the number of features to reduce embeding
        # size. NOTE however, that vocab cut-off is done based on
        # term frequency (might remove important but rare words (proper
        # nouns) from features
    )
    tfidf_vectors = vectorizer.fit_transform(
        node_attrs,
    )
    return tfidf_vectors


def load_entity_combi_graph():
    """ Return entity combi graph in a form usable with torch geometric.
    """

    # load node and edge tuples (with full, non-numeric features)
    node_tuples = _load_node_tuples(entities_only=True)
    # combi_edge: transform paper nodes into relationship between artifacts
    # (method, model, dataset, task)
    edge_tuples = _load_entity_combi_edge_tuples(
        final_node_set=set([ntup[0] for ntup in node_tuples]),
        scheme='weight'  # use weight scheme b/c it gives us a single integer
        #                  feature for edges rather than a variable length list
    )
    # build networkx graph for later conversion
    G = nx.Graph()
    # convert node features
    node_tuples_numerical = []
    node_id_numerical_map = {}
    node_type_map = {
        'dataset': 0,
        'method': 1,
        'model': 2,
        'task': 3
    }
    # # prepare LLM node description embeddings
    emb_dir = 'combigraph_mpnet-base_embeddings'
    idmap_fn = 'nodeid_to_fn.json'
    with open(os.path.join(emb_dir, idmap_fn)) as f:
        nodeid_to_fn = json.load(f)
    # # # prepare tfidf node description embeddings
    # G_lookup = nx.Graph()
    # G_lookup.add_nodes_from(node_tuples)
    # node_descrs = [
    #     _get_artifact_description(ntup[1], G_lookup)
    #     for ntup in node_tuples
    # ]
    # node_descr_vecs = _embed_string_atrs_tfidf(node_descrs)
    # # covert other feautures
    for new_node_id, ntup in enumerate(node_tuples):
        old_node_id = ntup[0]
        nattrs = ntup[1]
        node_id_numerical_map[old_node_id] = new_node_id  # assign numerical ID
        emb_fp = os.path.join(
            emb_dir,
            nodeid_to_fn[old_node_id]
        )
        descr_emb = np.load(emb_fp)
        attribs_num = {
            'id': new_node_id,  # needed as explicit attribute here?
            'type': node_type_map[nattrs['type']],
            'description': descr_emb
            # 'description': 0
            # 'description': node_descr_vecs[new_node_id].todense()
            # ^ currently fails with
            # AttributeError: 'matrix' object has no attribute 'dim'
            # see:
            # https://github.com/pyg-team/pytorch_geometric/issues/4445
            # https://github.com/pyg-team/pytorch_geometric/pull/4486

            # 'num_papers': nattrs.get('num_papers', -1)  # only meths & dsets
            # TODO: figure out/discuss how to handle different node types
            #       (i.e. a heterogeneous graph)
        }
        node_tuples_numerical.append(
            (new_node_id, attribs_num)
        )
    G.add_nodes_from(node_tuples_numerical)
    # convert edge features
    edge_tuples_numerical = []
    for etup in edge_tuples:
        attribs_num = {
            'weight': etup[2].get('weight', -1)
        }
        edge_tuples_numerical.append(
            (
                node_id_numerical_map[etup[0]],  # numerical node ID
                node_id_numerical_map[etup[1]],  # numerical node ID
                attribs_num
            )
        )
    G.add_edges_from(edge_tuples_numerical)
    data = from_networkx(
        G,
        group_node_attrs=['id', 'type', 'description'],
        group_edge_attrs=['weight']
    )
    return data
