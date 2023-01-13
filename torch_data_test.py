""" Test entity combi graph as PyTorch Geometric data
"""

import networkx as nx
from torch_geometric.utils.convert import from_networkx
from contextgraph.util.graph import _load_node_tuples,\
                                    _load_entity_combi_edge_tuples


node_tuples = _load_node_tuples(entities_only=True)
edge_tuples = _load_entity_combi_edge_tuples(
    final_node_set=set([ntup[0] for ntup in node_tuples]),
    scheme='weight'
)
G = nx.Graph()
node_tuples_numerical = []
for ntup in node_tuples:
    attribs_num = {
        'num_papers': ntup[1].get('num_papers', -1)
    }
    node_tuples_numerical.append(
        (ntup[0], attribs_num)
    )
G.add_nodes_from(node_tuples_numerical)
G.add_edges_from(edge_tuples)
data = from_networkx(
    G,
    group_node_attrs=['num_papers'],
    group_edge_attrs=['weight']
)
print(data)
