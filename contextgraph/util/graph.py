import csv
import os
import json
import networkx as nx
from contextgraph import config as cg_config


def _load_node_tuples(with_contexts=False):
    """ loads nodes as (<id>, <properties>) tuples
    """

    entity_tuples = []
    # first all regularly stored entities
    node_fns = [
        cg_config.graph_pprs_fn,
        cg_config.graph_meths_fn,
        cg_config.graph_dsets_fn,
        cg_config.graph_tasks_fn,
        cg_config.graph_modls_fn,
    ]
    if with_contexts:
        node_fns.append(cg_config.graph_cntxts_fn)
    for fn in node_fns:
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


def _load_edge_tuples(with_contexts=False, final_node_set=False):
    """ loads edges as (<id>, <properties>) tuples

        If final_node_set is given, only edges between existing nodes
        will be returned.
    """

    # first all regularly stored edges
    edge_types = [
        # used_in_paper
        [cg_config.graph_meths_to_pprs_fn, 'used_in_paper'],
        [cg_config.graph_dsets_to_pprs_fn, 'used_in_paper'],
        [cg_config.graph_tasks_to_pprs_fn, 'used_in_paper'],
        [cg_config.graph_modls_to_pprs_fn, 'used_in_paper'],
        # evaluated on
        [cg_config.graph_meths_to_dsets_fn, 'evaluated_on'],
        # # ^ NOTE: this means method used for (some task) on dataset
        # #         as extracted from evaltable
        # #         TODO: - mby also establish link between method and task
        # #               - or think of a "eval set" (meth, dset, task)
        # #               - should be associated w/ a date to prune graph of
        # #                 future
        # has task
        [cg_config.graph_dsets_to_tasks_fn, 'has_task'],
        # has subtask
        [cg_config.graph_tasks_to_subtasks_fn, 'has_subtask'],
        # # ^ could be reversed and given type 'part_of'
        # cites
        [cg_config.graph_ppr_to_ppr_fn, 'cites'],
        # part_of
        [cg_config.graph_meths_to_colls_fn, 'part_of']
        # (further down also: collection to area
        #                     entity     to context (if param set)
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
                if final_node_set is False or (
                        tail_id in final_node_set and
                        head_id in final_node_set
                        ):
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
                if final_node_set is False or (
                        tail_id in final_node_set and
                        head_id in final_node_set
                        ):
                    edge_tuples.append(
                        (
                            tail_id,
                            head_id,
                            {'type': 'part_of'}
                        )
                    )

    if with_contexts:
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
                if final_node_set is False or (
                        tail_id in final_node_set and
                        head_id in final_node_set
                        ):
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
                if final_node_set is False or (
                        tail_id in final_node_set and
                        head_id in final_node_set
                        ):
                    edge_tuples.append(
                        (
                            tail_id,
                            head_id,
                            {'type': 'part_of'}
                        )
                    )
    return edge_tuples


def _get_entity_coocurrence_edges(G, lim=-1):
    """ Determine pairs of entities (of dissimilar type) which
        are used in at least one common paper.

        To make subsequent processing steps more efficient,
        also determine all common papers for each entity
        pair as well as the earliest year in which any of those
        papers was published.
    """

    cooc_edges = dict()
    # get an undirected view to be able
    # to go from papers to used enitites
    uG = G.to_undirected(as_view=True)
    # for all papers
    lim_reached = False
    for (ppr_id, ppr_data) in uG.nodes(data=True):
        if lim_reached:
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
            if lim_reached:
                break
            e1 = G.nodes[e1_id]
            for e2_id in ppr_used_entities:
                if lim_reached:
                    break
                e2 = G.nodes[e2_id]
                # only need to check for dissimilar node type
                # ⇣ because dissimilar ID logically follows
                if e1['type'] != e2['type'] and \
                        not (e1['type'] == 'model' and e2['type'] == 'method'):
                    key = '_'.join(sorted([e1_id, e2_id]))
                    if key not in cooc_edges:
                        cooc_edges[key] = {
                            'edge': [e1_id, e2_id],
                            'cooc_pprs': set([ppr_id]),
                            'cooc_start_year': ppr_data['year'],
                            'cooc_start_month': ppr_data['month']
                        }
                    else:
                        cooc_edges[key]['cooc_pprs'].add(
                            ppr_id
                        )
                        cooc_edges[key]['cooc_start_year'] = min(
                            ppr_data['year'],
                            cooc_edges[key]['cooc_start_year']
                        )
                        cooc_edges[key]['cooc_start_month'] = min(
                            ppr_data['year'],
                            cooc_edges[key]['cooc_start_month']
                        )
                    if lim > 0 and len(cooc_edges) >= lim:
                        lim_reached = True

    return cooc_edges  # 2M edges if lim is not set


def _get_two_hop_pair_neighborhood_nodes(cooc_entity_pair, G):
    """ Get two hop neighborhood for a pair of entities
        that co-occur in at least one paper.
    """

    # TODO: generalize to n hops

    keep_node_ids = set()

    # get an undirected view to be able
    # to go from papers to used enitites
    uG = G.to_undirected(as_view=True)

    # for both entities
    for entity in cooc_entity_pair['edge']:
        # first hop neighborhood
        for one_hop_neigh in uG.neighbors(entity):
            keep_node_ids.add(one_hop_neigh)
            # second hop neighborhood
            for two_hop_neigh in uG.neighbors(one_hop_neigh):
                keep_node_ids.add(two_hop_neigh)
    return G.subgraph(keep_node_ids)


def _get_pruned_graph(cooc_entity_pair, G):
    """ For a pair of entities that co-occur in at least one paper
        (given as a dictionary containing the entities and a list of
        all co-occurrence papers)
        return a pruned version of the graph G that contains only
        papers before the first co-occurrence paper.
    """

    # TODO: currently takes 4 seconds on full graph -> too long?
    #       (takes 126 ms on an edge’s two hop neighborhood)

    keep_node_ids = []

    # determine all papers published after earliest cooc ppr
    for node_id in G.nodes:
        node_data = G.nodes[node_id]
        if len(node_data) > 0:
            if node_data['type'] != 'paper':
                # not a paper, keep
                # (TODO: also consider other enitites
                #  that should be removed)
                keep_node_ids.append(node_id)
            elif node_data['year'] < cooc_entity_pair['cooc_start_year'] and \
                    node_data['month'] < cooc_entity_pair['cooc_start_month']:
                # a paper but published early enough
                keep_node_ids.append(node_id)

    return G.subgraph(keep_node_ids)


def get_pair_graphs(n_true_pairs, G):
    """ Return 2 × n_true_pairs graphs with their respective prediction edge.
            - half are *prunded* graphs of co-occurring entities
            - the other half are *prunded* graphs of non-co-occurring entities

        A pair of co-occurring entities are two differently typed entities
        which have at least one common paper in which they are used.
    """

    # get positive training examples
    true_pair_grahps = []
    # don’t apply limit here                       |
    # b/c it’s fast enough to do the whole graph   V
    cooc_edges = _get_entity_coocurrence_edges(G, -1)
    # generate negative training examples
    # # cluster co-occurrence edges by year of earliest cooc ppr
    edge_year_clusters = dict()
    for key, cooc_edge in cooc_edges.items():
        # only use year here and not also month
        if cooc_edge['cooc_start_year'] not in edge_year_clusters:
            edge_year_clusters[cooc_edge['cooc_start_year']] = []
        edge_year_clusters[cooc_edge['cooc_start_year']].append(
            cooc_edge
        )
    # # generate corrupted co-occurrence edges by swapping entities
    # # between edges in the same co-occurrence year cluster that
    # # have disjoint co-occurrence paper sets
    for cooc_start_year, cooc_edge_list in edge_year_clusters.items():
        print(cooc_start_year)
        print(len(cooc_edge_list))
        # TODO take year distribution stratified sample s.t. corrent
        #      number of positive and negative training examples can
        #      be returned
    return
    for key, cooc_edge in cooc_edges.items():
        # reduce to neighborhood that is potentially necessary (for speedup)
        neigh_G = _get_two_hop_pair_neighborhood_nodes(cooc_edge, G)
        # remove paper nodes based on time constraint
        pruned_G = _get_pruned_graph(cooc_edge, neigh_G)
        # reduce to neighborhood (gets rid of stuff that is onyl
        # connected in unpruned graph)
        pruned_neigh_G = _get_two_hop_pair_neighborhood_nodes(
            cooc_edge,
            pruned_G
        )
        true_pair_grahps.append({
            'prediction_edge': cooc_edge,
            'graph': pruned_neigh_G
        })
    return true_pair_grahps


def make_shallow(G):
    """ Remove all attributes except for type from nodes and edges
    """

    if type(G) == nx.DiGraph:
        shallow_G = nx.DiGraph()
    else:
        shallow_G = nx.Graph()

    for nid, ndata in G.nodes.items():
        shallow_G.add_nodes_from([(nid, {'type': ndata['type']})])
    for edge_tuple, edata in G.edges.items():
        naid, nbid = edge_tuple
        shallow_G.add_edges_from([(naid, nbid, {'type': edata['type']})])

    return shallow_G


def load_graph(shallow=False, directed=True, with_contexts=False):
    """ Load nodes and edges into a NetworkX digraph.

        If shallow is True, all node and edge features
        except for type will be discarded.
    """

    node_tuples = _load_node_tuples(
        with_contexts=with_contexts
    )
    edge_tuples = _load_edge_tuples(
        with_contexts=with_contexts,
        # make sure not to give networkx a reason to implicitly
        # add empty, untyped nodes because of edges
        final_node_set=set([ntup[0] for ntup in node_tuples])
    )
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
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from(node_tuples)
    G.add_edges_from(edge_tuples)
    return G
