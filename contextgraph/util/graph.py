import csv
import os
import json
import networkx as nx
from contextgraph import config as cg_config


def _load_node_tuples():
    """ loads nodes as (<id>, <properties>) tuples
    """

    entity_tuples = []
    # first all regularly stored entities
    for fn in [
        cg_config.graph_pprs_fn,
        cg_config.graph_meths_fn,
        cg_config.graph_dsets_fn,
        cg_config.graph_tasks_fn,
        cg_config.graph_modls_fn,
        cg_config.graph_cntxts_fn
    ]:
        with open(os.path.join(cg_config.graph_data_dir, fn)) as f:
            for line in f:
                entity = json.loads(line)
                entity_tuples.append((entity['id'], entity))
    # then some special processing for the areas to collections data
    with open(os.path.join(
        cg_config.graph_data_dir,
        cg_config.graph_meth_areas_fn
    )) as f:
        for line in f:
            # area
            area = json.loads(line)
            entity_tuples.append((
                area['id'],
                {
                    'id': area['id'],
                    'type': area['type'],
                    'name': area['name']
                }
            ))
            # all contained collections
            for coll in area['collections']:
                entity_tuples.append(
                    (coll['id'], coll)
                )
    return entity_tuples


def _load_edge_tuples():
    """ loads edges as (<id>, <properties>) tuples
    """

    # first all regularly stored edges
    edge_types = [
        # used_in_paper
        [cg_config.graph_meths_to_pprs_fn, 'used_in_paper'],
        [cg_config.graph_dsets_to_pprs_fn, 'used_in_paper'],
        [cg_config.graph_tasks_to_pprs_fn, 'used_in_paper'],
        [cg_config.graph_modls_to_pprs_fn, 'used_in_paper'],
        # used_together (tread as symmetrical?)
        [cg_config.graph_meths_to_dsets_fn, 'used_together'],
        [cg_config.graph_dsets_to_tasks_fn, 'used_together'],
        # cites
        [cg_config.graph_ppr_to_ppr_fn, 'cites'],
        # part_of
        [cg_config.graph_meths_to_colls_fn, 'part_of'],
        [cg_config.graph_tasks_to_subtasks_fn, 'part_of']
        # (further down also: collection to area
        #                     entity     to context
        #                     context    to paper)
    ]
    edge_tuples = []
    for (fn, edge_type) in edge_types:
        header_idxs = [0, 1]
        if edge_type == 'cites':
            header_idxs = [3, 4]
        with open(os.path.join(cg_config.graph_data_dir, fn)) as f:
            csv_reader = csv.DictReader(
                f,
                delimiter=',',
                quoting=csv.QUOTE_NONE
            )
            headers = csv_reader.fieldnames
            for row in csv_reader:
                tail_id = row[headers[header_idxs[0]]]
                head_id = row[headers[header_idxs[1]]]
                edge_tuples.append(
                    (
                        tail_id,
                        head_id,
                        {'type': edge_type}
                    )
                )
    # then some special processing for the areas to collections data
    with open(os.path.join(
        cg_config.graph_data_dir,
        cg_config.graph_meth_areas_fn
    )) as f:
        for line in f:
            area = json.loads(line)
            head_id = area['id']
            for coll in area['collections']:
                tail_id = coll['id']
                edge_tuples.append(
                    (
                        tail_id,
                        head_id,
                        {'type': 'part_of'}
                    )
                )
    # lastly some special processing for contexts to papers and entities
    with open(os.path.join(
        cg_config.graph_data_dir,
        cg_config.graph_cntxts_fn
    )) as f:
        for line in f:
            cntxt = json.loads(line)
            # entity to context
            tail_id = cntxt['entity_id']
            head_id = cntxt['id']
            edge_tuples.append(
                (
                    tail_id,
                    head_id,
                    {'type': 'part_of'}
                )
                )
            # context to paper
            tail_id = cntxt['id']
            head_id = cntxt['paper_pwc_id']
            edge_tuples.append(
                (
                    tail_id,
                    head_id,
                    {'type': 'part_of'}
                )
                )
    return edge_tuples


def load_graph():
    node_tuples = _load_node_tuples()
    edge_tuples = _load_edge_tuples()
    G = nx.DiGraph()
    G.add_nodes_from(node_tuples)
    G.add_edges_from(edge_tuples)
    return G
