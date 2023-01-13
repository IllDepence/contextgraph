import csv
import os
import json
import random
import networkx as nx
import numpy as np
from contextgraph import config as cg_config


class cooc_edge_dict(dict):
    def __hash__(self):
            return hash(tuple(sorted(self['edge'])))


def _load_node_tuples(with_contexts=False, entities_only=False):
    """ loads nodes as (<id>, <properties>) tuples
    """

    entity_tuples = []
    # first all regularly stored entities
    node_fns = [
        cg_config.graph_meths_fn,
        cg_config.graph_dsets_fn,
        cg_config.graph_tasks_fn,
        cg_config.graph_modls_fn,
    ]
    if not entities_only:
        node_fns.append(cg_config.graph_pprs_fn)
    if with_contexts:
        node_fns.append(cg_config.graph_cntxts_fn)
    for fn in node_fns:
        with open(os.path.join(cg_config.graph_data_dir, fn)) as f:
            for line in f:
                entity = json.loads(line)
                entity_tuples.append((entity['id'], entity))
    if entities_only:
        # already done, b/c areas and collections will not be added
        return entity_tuples
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
                        cooc_edges[key] = cooc_edge_dict({
                            'edge': [e1_id, e2_id],
                            'cooc_pprs': set([ppr_id]),
                            'cooc_start_year': ppr_data['year'],
                            'cooc_start_month': ppr_data['month']
                        })
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


def _corrupted_cooc_eges(e1, e2, G):
    """ Return corrupted co-occurrence edges by
        swapping tail nodes.

        Assumes that
            - earliest co-occurrence paper of e1 and e1
              were published in the same year.
            - sets co-occurrence papers of e1 and e1 are
              disjoint.
    """

    shared_cooc_start_year = e1['cooc_start_year']
    # the start month of negative samples has a *LARGE*
    # impact on the resulting size of pruned graphs,
    # as many papers are published in a short time.
    # tried
    # - default 1 -> neg smpl graphs way smaller
    # - default 6 -> "
    # - for each corrupted edge take the
    #   month of the entity which is more
    #   connected -> "
    # - same as above but never
    #   less than 7 -> looks about right
    # - default 12 -> neg smpl graphs somewhat larger
    # (clustering by year and month beforehand leads
    #  to average cluster sizes of mean ~2.9 and
    #  median 2 => swapping partners are not really
    #  random anymore but determined by publishing month
    #  => likely underirable)
    corr1_edge = [e1['edge'][0], e2['edge'][1]]
    if len(G.adj[e1['edge'][0]]) > len(G.adj[e2['edge'][1]]):
        corr1_month = e1['cooc_start_month']
    else:
        corr1_month = e2['cooc_start_month']
    corr1 = cooc_edge_dict({
        'edge': corr1_edge,
        'cooc_pprs': set(),  # empty
        'cooc_start_year': shared_cooc_start_year,
        'cooc_start_month': max(corr1_month, 7),
    })
    corr2_edge = [e2['edge'][0], e1['edge'][1]]
    if len(G.adj[e2['edge'][0]]) > len(G.adj[e1['edge'][1]]):
        corr2_month = e2['cooc_start_month']
    else:
        corr2_month = e1['cooc_start_month']
    corr2 = cooc_edge_dict({
        'edge': corr2_edge,
        'cooc_pprs': set(),  # empty
        'cooc_start_year': shared_cooc_start_year,
        'cooc_start_month': max(corr2_month, 7),
    })
    return [corr1, corr2]


def get_pair_graphs(n_true_pairs, G):
    """ Return 2 × n_true_pairs graphs with their respective prediction edge.
            - half are *prunded* graphs of co-occurring entities
            - the other half are *prunded* graphs of non-co-occurring entities

        A pair of co-occurring entities are two differently typed entities
        which have at least one common paper in which they are used.
    """

    # get positive training examples
    # # don’t apply limit here                          |
    # # b/c it’s fast enough to do the whole graph      V
    cooc_edges_full = _get_entity_coocurrence_edges(G, -1)
    # generate negative training examples
    # # cluster co-occurrence edges by year of earliest cooc ppr
    edge_year_clusters = dict()
    for key, cooc_edge in cooc_edges_full.items():
        # only use year here and not also month
        if cooc_edge['cooc_start_year'] not in edge_year_clusters:
            edge_year_clusters[cooc_edge['cooc_start_year']] = []
        edge_year_clusters[cooc_edge['cooc_start_year']].append(
            cooc_edge
        )
    # # generate corrupted co-occurrence edges by swapping entities
    # # between edges in the same co-occurrence year cluster that
    # # have disjoint co-occurrence paper sets
    num_cooc_edges_full = sum([len(l) for k, l in edge_year_clusters.items()])
    year_smpl_sizes = dict()
    for cooc_start_year, cooc_edge_list in edge_year_clusters.items():
        # determine co-occurrence year cluter characteristics
        proportion = len(cooc_edge_list)/num_cooc_edges_full
        if n_true_pairs > 0:
            sample_size = round(n_true_pairs * proportion)
        else:
            sample_size = round(len(cooc_edge_list) * proportion)
        year_smpl_sizes[cooc_start_year] = sample_size
    if n_true_pairs > 0:
        # # make sure year samples add up to n_true_pairs
        smpl_diff = n_true_pairs - sum(year_smpl_sizes.values())
        fill_year = list(year_smpl_sizes.keys())[
            np.argmax(year_smpl_sizes.values())
        ]
        year_smpl_sizes[fill_year] += smpl_diff
    # # sample positive and negative prediction edges
    cooc_edges_pos = set()
    cooc_edges_neg = set()
    for cooc_start_year, cooc_edge_list in edge_year_clusters.items():
        sample_size = year_smpl_sizes[cooc_start_year]
        if sample_size < 1:
            continue
        # create negative samples
        shuf1 = random.sample(cooc_edge_list, len(cooc_edge_list))
        shuf2 = random.sample(cooc_edge_list, len(cooc_edge_list))
        size_reached = False
        year_smpl_pos = set()
        year_smpl_neg = set()
        for cooc_edge1 in shuf1:
            if size_reached:
                break
            for cooc_edge2 in shuf2:
                if len(year_smpl_pos) >= sample_size and \
                   len(year_smpl_neg) >= sample_size:
                    size_reached = True
                if len(set.intersection(
                    cooc_edge1['cooc_pprs'],
                    cooc_edge2['cooc_pprs']
                )) == 0:
                    # true prediction edges
                    year_smpl_pos.add(cooc_edge1)
                    year_smpl_pos.add(cooc_edge2)
                    # false prediction edges
                    corr1, corr2 = _corrupted_cooc_eges(
                        cooc_edge1, cooc_edge2, G
                    )
                    year_smpl_neg.add(corr1)
                    year_smpl_neg.add(corr2)
        # cut year’s contribution to full sample to size
        cooc_edges_pos.update(
            random.sample(
                year_smpl_pos,
                min(sample_size, len(year_smpl_pos))
                )
        )
        cooc_edges_neg.update(
            random.sample(
                year_smpl_neg,
                min(sample_size, len(year_smpl_neg))
                )
        )
    # create graphs
    true_pair_grahps = []
    false_pair_grahps = []
    for graph_list, edge_list in [
        (true_pair_grahps, cooc_edges_pos),
        (false_pair_grahps, cooc_edges_neg)
    ]:
        for cooc_edge in edge_list:
            # reduce to neighborhood that is potentially necessary (speedup)
            neigh_G = _get_two_hop_pair_neighborhood_nodes(cooc_edge, G)
            # remove paper nodes based on time constraint
            pruned_G = _get_pruned_graph(cooc_edge, neigh_G)
            # reduce to neighborhood (gets rid of stuff that is onyl
            # connected in unpruned graph)
            pruned_neigh_G = _get_two_hop_pair_neighborhood_nodes(
                cooc_edge,
                pruned_G
            )
            graph_list.append({
                'prediction_edge': cooc_edge,
                'graph': pruned_neigh_G
            })
    # test for n_true_pairs = 200:
    #
    # In [204]: np.mean([len(x['graph'].nodes) for x in fls])
    # Out[204]: 134.485
    # In [205]: np.mean([len(x['graph'].nodes) for x in tru])
    # Out[205]: 2742.075
    # -> why fewer nodes for negative edges?
    # TODO: debug
    #
    # In [202]: np.mean([len(x['graph'].edges) for x in fls])
    # Out[202]: 193.685
    # In [203]: np.mean([len(x['graph'].edges) for x in tru])
    # Out[203]: 17480.44
    return true_pair_grahps, false_pair_grahps


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


def _load_entity_combi_edge_tuples(final_node_set=False, scheme='sequence'):
    """ Determine edges tuples that represent the
        number of co-occurrence papers between entities.

        schemes:
            - sequence: time sequence of co-occurences
            - weight: number of co-occurences
    """

    # not the most time efficient, but simply
    # go from the full graph using existing code
    G = load_full_graph()
    cooc_edges_full = _get_entity_coocurrence_edges(G, -1)
    # create combi edges
    edge_tuples = []
    # determine earliest cooc ppr in current data is from 1994
    cooc_years = []
    for k, ce in cooc_edges_full.items():
        cooc_years.append(ce['cooc_start_year'])
    beginning_of_time = min(cooc_years)
    for key, cooc_edge in cooc_edges_full.items():
        ppr_a_id = cooc_edge['edge'][0]
        ppr_b_id = cooc_edge['edge'][1]
        if scheme == 'sequence':
            # build co-occurrence time sequence
            cooc_time_sequence = []
            cooc_ppr_sequence = []
            for ppr_id in cooc_edge['cooc_pprs']:
                y = G.nodes[ppr_id]['year']
                m = G.nodes[ppr_id]['month']
                # there is also day info, but mby not so relevant
                if y > 0 and m > 0:  # is -1 if info not given
                    year_in_cooc_time = y - beginning_of_time
                    month_in_cooc_time = (year_in_cooc_time * 12) + m
                    cooc_time_sequence.append(month_in_cooc_time)
                    cooc_ppr_sequence.append(ppr_id)
            edge_property = {
                'interaction_sequence': cooc_time_sequence,
                'linker_sequence': cooc_ppr_sequence
            }
        elif scheme == 'weight':
            # TODO:
            # consider then using add_weighted_edges_from later
            edge_property = {'weight': len(cooc_edge['cooc_pprs'])}
        if not final_node_set or \
                (final_node_set and
                 ppr_a_id in final_node_set and
                 ppr_b_id in final_node_set):
            edge_tuples.append(
                (
                    ppr_a_id,
                    ppr_b_id,
                    edge_property
                )
            )
    return edge_tuples


def load_entity_combi_graph(scheme='sequence'):
    """ Load graph only containing the entity nodes,
        connected by weighted edges that represent the
        number of co-occurrence papers.

        NOTE: to access edge attributes later,
               G.edges(data=True) has to be used!
    """

    node_tuples = _load_node_tuples(entities_only=True)
    edge_tuples = _load_entity_combi_edge_tuples(
        # make sure not to give networkx a reason to implicitly
        # add empty, untyped nodes because of edges
        final_node_set=set([ntup[0] for ntup in node_tuples]),
        scheme=scheme
    )
    G = nx.Graph()
    G.add_nodes_from(node_tuples)
    G.add_edges_from(edge_tuples)
    return G


def load_full_graph(shallow=False, directed=True, with_contexts=False):
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
