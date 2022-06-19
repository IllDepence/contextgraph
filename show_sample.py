import json
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# TODO
# - position prediction edge nodes when layouting
# - look for alternative layouting ways

base_bath = '../data_tmp/sc_graph_samples'
fn_base = 'pair_graph_sample_312'
fn = f'{fn_base}_graph.json'
with open(os.path.join(base_bath, fn)) as f:
    G = nx.cytoscape_graph(json.load(f))
fn = f'{fn_base}_prediction_edge.json'
with open(os.path.join(base_bath, fn)) as f:
    prediction_edge = json.load(f)

print(nx.info(G))
# pos = nx.spring_layout(
#     G,
#     k=1/np.sqrt(len(G.nodes)),
#     iterations=200  # detault is 50
# )
# pos[prediction_edge['edge'][0]][0] -= 1.5
# pos[prediction_edge['edge'][1]][0] += 1.5

pos = nx.kamada_kawai_layout(G)
# TODO: test different dist values

# pos = nx.spectral_layout(G)

node_color_list = []
node_size_list = []
for node_id, node_data in G.nodes.items():
    s = 150
    if node_id in prediction_edge['edge']:
        s = 1000
    c = '#8f8f8f'
    if node_data['type'] == 'paper':
        c = '#7289ab'  # blueish gray
    if node_data['type'] == 'dataset':
        c = '#3bbf2a'  # green
    if node_data['type'] == 'method':
        c = '#5047c4'  # blue
    if node_data['type'] == 'task':
        c = '#c45547'  # red
    if node_data['type'] == 'model':
        c = '#a847c4'  # violet
    node_color_list.append(c)
    node_size_list.append(s)
edge_color_list = []
for edge_id, edge_data in G.edges.items():
    # TODO:
    # highlight shortest path(s) between prediction edge nodes
    c = '#8f8f8f'
    if edge_data['type'] == 'used_in_paper':
        c = '#0b41b0'  # blue
    if edge_data['type'] == 'evaluated_on':
        c = '#a6b00b'  # occer
    if edge_data['type'] == 'cites':
        c = '#bd2222'  # red
    edge_color_list.append(c)
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=node_color_list,
    edge_color=edge_color_list,
    node_size=node_size_list,
    font_size=8,
    font_color=(0.7, 0.7, 0.7, 0.5)
)
nx.draw_networkx_nodes
plt.show()
