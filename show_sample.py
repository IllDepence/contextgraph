import json
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# TODO
# - color edges
# - position prediction edge nodes when layouting
# - look for alternative layouting ways

base_bath = '../data_tmp/sc_graph_samples'
fn_base = 'pair_graph_sample_312'
fn = f'{fn_base}_graph.json'
with open(os.path.join(base_bath, fn)) as f:
    G = nx.cytoscape_graph(json.load(f))

print(nx.info(G))
pos = nx.spring_layout(
    G,
    k=1/np.sqrt(len(G.nodes)),
    iterations=200  # detault is 50
)
node_color_list = []
for node_id, node_data in G.nodes.items():
    c = '#8f8f8f'
    if node_data['type'] == 'paper':
        c = '#4d76b3'
    if node_data['type'] == 'dataset':
        c = '#3bbf2a'
    if node_data['type'] == 'method':
        c = '#5047c4'
    if node_data['type'] == 'task':
        c = '#c45547'
    if node_data['type'] == 'model':
        c = '#a847c4'
    node_color_list.append(c)
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=node_color_list,
    edge_color=(0.7, 0.7, 0.7, 0.5),
    # node_size=1500,
    # node_color='yellow',
    font_size=8,
    font_color=(0.7, 0.7, 0.7, 0.5)
)
plt.show()
