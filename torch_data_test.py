""" Demo of loading entity combi graph as PyTorch Geometric data
"""

from contextgraph.util.torch import load_entity_combi_graph

data = load_entity_combi_graph()
print(data)
