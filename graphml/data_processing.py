import warnings
import numpy as np

def prepare_samples(param_object):

    file_pairs = []
    numbers = ["0" + str(i) for i in range(10)] + \
              [str(i) for i in range(10, param_object.NUM_SAMPLES_PER_LABEL)]
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
