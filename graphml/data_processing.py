import warnings
import numpy as np
import os
import json
import networkx as nx

def prepare_samples(param_object):

    file_pairs = []
    # numbers = ["0" + str(i) for i in range(10)] + \
    #           [str(i) for i in range(10, param_object.NUM_SAMPLES_PER_LABEL)]
    # numbers = [str(i) for i in range(6500, 7000)]
    numbers = [str(i) for i in range(0, 3)]
    labels = ["pos", "neg"]

    for num in numbers:
        for l in labels:
            file_graph = "pair_graph_sample_" + l +"_" + num +"_graph.json"
            file_pred = "pair_graph_sample_" + l +"_" + num +"_prediction_edge.json"
            file_pairs.append((file_graph, file_pred, l))
    return file_pairs


def process_name(node_pair_to_predict):
    '''get rid of "pwc" and return the concatenation of two node names'''
    node_names = [node.split(":")[-1].replace("/", "_") \
                  for node in node_pair_to_predict]
    final_name = node_names[0] + "-" + node_names[1]
    return final_name


def operate(df_2r, pattern):

    if pattern == "avg":
        return df_2r.sum()
    elif pattern == "hadamard":
        return df_2r.iloc[0, :] * df_2r.iloc[1, :0]
    elif pattern == "l1":
        return np.abs(df_2r.iloc[0, :] - df_2r.iloc[1, :])
    elif pattern == "l2":
        return (df_2r.iloc[0, :] - df_2r.iloc[1, :]) ** 2
    else:
        message = 'A unsupported pattern for processing embeddings of \
                    two target nodes has been given, please use a correct \
                    pattern from either "avg", "hadamard", "l1" or "l2"'
        warnings.warn(message)

def generate_atom_graph(file_dir, file_graph, file_pred,
                         directed=True, export=False):

    file_graph_path = os.path.join(file_dir, file_graph)
    file_pred_path = os.path.join(file_dir, file_pred)
    with open(file_graph_path, "r") as g:
        data = json.load(g)
    with open(file_pred_path, "r") as p:
        predicting = json.load(p)
    node_pair_to_predict = predicting["edge"]
    elements = data.get("elements", {})

    nodes = []
    pairs = []
    for node in elements["nodes"]:
        info = node["data"]
        nodes.append((info["id"],
                      {"type": info["type"]}))
    for edge in elements["edges"]:
        structure = edge["data"]
        pairs.append((structure["source"],
                      structure["target"],
                      {"type": structure["type"]}))
    if directed:
        atom_graph = nx.DiGraph()
    else:
        atom_graph = nx.Graph()
    atom_graph.add_nodes_from(nodes)
    atom_graph.add_edges_from(pairs)

    if export:
        final_name = process_name(node_pair_to_predict)
        # cooc_pprs = predicting["cooc_pprs"]
        nx.write_graphml(atom_graph, final_name)

    return atom_graph, node_pair_to_predict
