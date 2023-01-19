""" Methods for loading the graph for pytorch gemoetric
"""

import networkx as nx
from torch_geometric.utils.convert import from_networkx
from contextgraph.util.graph import _load_node_tuples,\
                                    _load_entity_combi_edge_tuples


def load_full_graph():
    """ Return full graph in a form usable with torch geometric.
    """

    raise NotImplementedError


def load_entity_combi_graph():
    """ Return entity combi graph in a form usable with torch geometric.
    """

    # load node and edge tuples (with full, non-numeric features)
    node_tuples = _load_node_tuples(entities_only=True)
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
    for new_node_id, ntup in enumerate(node_tuples):
        old_node_id = ntup[0]
        nattrs = ntup[1]
        node_id_numerical_map[old_node_id] = new_node_id  # assign numerical ID
        attribs_num = {
            'id': new_node_id,  # needed as explicit attribute here?
            'type': node_type_map[nattrs['type']],
            'description': 0,  # TODO: implement conversion
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
