""" Demo of loading CoCon as a NetworkX graph or PyTorch Geometric data
"""

from contextgraph.util.graph import load_full_graph
from contextgraph.util.torch import load_entity_combi_graph

# NetworkX, full graph
G = load_full_graph()
print(type(G))

# PyTorch Geometric, simple graph
data = load_entity_combi_graph()
print(type(data))
