import networkx as nx
from node2vec import Node2Vec
import os
import json
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm


class Node2VecEmbedder():

    def __init__(self, param_object, pattern="avg"):
        '''
        param_object: a object from dataclass that stores all necessary hyperparameter for Node2Vec
        pattern: the pattern of how the final embedding of two embeddings of targets nodes are calculated
        '''

        self.param = param_object
        self.pattern = pattern
        self.embeddings = None

    def _prepare_samples(self):

        file_pairs = []
        numbers = ["0" + str(i) for i in range(10)] + \
                  [str(i) for i in range(10, self.param.NUM_SAMPLES_PER_LABEL)]
        labels = ["pos", "neg"]

        for num in numbers:
            for l in labels:
                file_graph = "pair_graph_sample_"+l+"_"+num+"_graph.json"
                file_pred = "pair_graph_sample_"+l+"_"+num+"_prediction_edge.json"
                file_pairs.append((file_graph, file_pred, l))
        return file_pairs

    def _generate_atom_graph(self, file_dir, file_graph, file_pred,
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
            final_name = self._process_name(node_pair_to_predict)
            # cooc_pprs = predicting["cooc_pprs"]
            nx.write_graphml(atom_graph, final_name)

        return atom_graph, node_pair_to_predict

    @staticmethod
    def _process_name(node_pair_to_predict):
        '''get rid of "pwc" and return the concatenation of two node names'''
        node_names = [node.split(":")[-1].replace("/", "_") \
                      for node in node_pair_to_predict]
        final_name = node_names[0] + "-" + node_names[1]
        return final_name

    @staticmethod
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

    def node_embedding(self, directory, directed=True, export_each_graph=False):

        file_pairs = self._prepare_samples()
        df_emb = pd.DataFrame(columns=range(self.param.DIMENSIONS))
        df_label = pd.DataFrame(columns=["label"])

        for file_pair in tqdm(file_pairs, desc="processing file pairs:"):
            file_graph = file_pair[0]
            file_pred = file_pair[1]
            label = file_pair[2]
            graph, node_pair_to_predict = self._generate_atom_graph(file_dir=directory,
                                                                   file_graph=file_graph,
                                                                   file_pred=file_pred,
                                                                   directed=directed,
                                                                  export=export_each_graph)
            # skip the empty graphs
            if len(graph.nodes) == 0:
                continue
            node2vec = Node2Vec(graph,
                                dimensions=self.param.DIMENSIONS,
                                walk_length=self.param.WALK_LENGTH,
                                num_walks=self.param.NUM_WALKS,
                                seed=self.param.SEED
                                )
            model = node2vec.fit(
                window=self.param.WINDOW,
                min_count=self.param.MIN_COUNT,
                batch_words=self.param.BATCH_WORDS
            )
            embeddings = pd.DataFrame(
                [model.wv.get_vector(str(n)) for n in graph.nodes()],
                index=graph.nodes
            )
            try:
                emb_target_nodes = embeddings.loc[node_pair_to_predict, :]
                emb_serie = self.operate(emb_target_nodes, self.pattern)
            except:
                emb_serie = pd.Series(index=range(128), dtype=np.float64)

            final_name = self._process_name(node_pair_to_predict)
            df_emb.loc[final_name, :] = emb_serie
            df_label.loc[final_name, :] = label

        df_ml = pd.concat([df_emb, df_label], axis=1)
        if df_emb.isnull().sum().sum() > 0:
            message = "targets node in prediction files are not consistent " \
                      "with nodes in graph file, check dataframe manully to " \
                      "get more information"
            warnings.warn(message)
        self.embeddings = df_ml

        return df_ml
