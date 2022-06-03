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


def _get_entity_coocurrence_edges(G):
    """ Determine pairs of entities (of dissimilar type) which
        are used in at least one common paper.

        To make subsequent processing steps more efficient,
        also determine to earliest common paper for each entity
        pair.
    """

    cooc_edges = dict()
    # get an undirected view to be able
    # to go from papers to used enitites
    uG = G.to_undirected(as_view=True)
    # for all papers
    i = 0
    for (ppr_id, ppr_data) in uG.nodes(data=True):
        i += 1
        if i > 1000:
            break
        if 'type' not in ppr_data:
            # nodes without any data get created by
            # adding edges to inexistent nodes
            continue
        if ppr_data['type'] != 'paper':
            continue
        ppr_used_entities = set()
        # for all their used entities
        for (entity_id, edge_data) in uG.adj[ppr_id].items():
            if edge_data['type'] != 'used_in_paper':
                continue
            ppr_used_entities.add(entity_id)
        # get pairs of entities of dissimilar type
        for e1_id in ppr_used_entities:
            e1 = G.nodes[e1_id]
            for e2_id in ppr_used_entities:
                e2 = G.nodes[e2_id]
                # only need to check for dissimilar node type
                # ⇣ because dissimilar ID logically follows
                if e1['type'] != e2['type']:
                    key = '_'.join(sorted([e1['id'], e2['id']]))
                    if key not in cooc_edges:
                        cooc_edges[key] = {
                            'edge': [e1_id, e2_id],
                            'cooc_pprs': set([ppr_id])
                        }
                    else:
                        cooc_edges[key]['cooc_pprs'].add(
                            ppr_id
                        )
    # Usage example
    # cooc_edges = _get_entity_coocurrence_edges(G)
    # for k, ce in cooc_edges.items():
    #     if len(ce['cooc_pprs']) > 1:
    #         print('---'.join(ce['edge']))
    #         for ppr_id in ce['cooc_pprs']:
    #             ppr = G.nodes[ppr_id]
    #             print('\t', ppr_id, ppr['date'])
    #
    # TODO: - use sample to prune graph according to eariest cooc ppr
    #       - visually inspect pruned graphs (mby w/ limited neighborhoods
    #         of entitiy pair)
    return cooc_edges  # currently 2M edges


def load_graph(shallow=False):
    """ Load nodes and edges into a NetworkX digraph.

        If shallow is True, all node and edge features
        except for type will be discarded.
    """

    node_tuples = _load_node_tuples()
    edge_tuples = _load_edge_tuples()
    if shallow:
        shallow_node_tuples = []
        shallow_edge_tuples = []
        for n in node_tuples:
            shallow_node_tuples.append((
                n[0],
                {'type': n[1]['type']}
            ))
        for e in edge_tuples:
            shallow_edge_tuples.append((
                e[0],
                e[1],
                {'type': e[2]['type']}
            ))
        node_tuples = shallow_node_tuples
        edge_tuples = shallow_edge_tuples
    G = nx.DiGraph()
    G.add_nodes_from(node_tuples)
    G.add_edges_from(edge_tuples)
    return G
