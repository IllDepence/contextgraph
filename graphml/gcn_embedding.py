import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from stellargraph.layer import GCN
from stellargraph import StellarGraph
from stellargraph.mapper import FullBatchNodeGenerator
from data_processing import prepare_samples, process_name, operate, generate_atom_graph


class GCNEmbedder():

    def __init__(self, param_object):
        '''
        param_object: a object from dataclass that stores all necessary hyperparameter for Node2Vec
        pattern: the pattern of how the final embedding of two embeddings of targets nodes are calculated
        '''

        self.param = param_object
        self.pattern = self.param.PATTERN
        self.embeddings = None

    def create_stellargraph(self, graph):
        df = pd.DataFrame(columns=["node", "type"])
        df["node"] = [node[0] for node in graph.nodes(data=True)]
        # TODO: deal with potential missing type, right now simply set as "task"
        df["type"] = [node[-1].get("type", "task") for node in graph.nodes(data=True)]
        df = df.set_index("node")
        enc = OneHotEncoder(handle_unknown='ignore')
        vec_onehot = enc.fit_transform(df).toarray()

        features_dict = dict()
        for row in pd.DataFrame(vec_onehot, index=df.index, columns=enc.categories_).iterrows():
            features_dict[row[0]] = list(row[1])
        nx.set_node_attributes(graph, features_dict, "feature")
        stellargraph = StellarGraph.from_networkx(graph,
                                                  node_features="feature",
                                                  edge_type_attr="type",
                                                  node_type_default="task"
                                                  )
        return stellargraph, vec_onehot

    def gcn_learn(self, stellargraph, training_labels):
        nodes = list(stellargraph.nodes())
        generator = FullBatchNodeGenerator(stellargraph, method="gcn")
        train_gen = generator.flow(nodes, training_labels)
        gcn = GCN(layer_sizes=[self.param.DIMENSIONS, self.param.DIMENSIONS],
                  activations=["relu", "relu"],
                  generator=generator,
                  dropout=0.5
                  )
        x_inp, x_out = gcn.in_out_tensors()
        predictions = tf.keras.layers.Dense(units=training_labels.shape[1], activation="softmax")(x_out)
        model = tf.keras.Model(inputs=x_inp, outputs=predictions)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.param.LEARNING_RATE_GCN),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=["acc"]
                      )
        _ = model.fit(train_gen,
                      epochs=self.param.EPOCHS_GCN,
                      verbose=0,
                      shuffle=False
                    )
        embedding_model = tf.keras.Model(inputs=x_inp, outputs=x_out)
        emb = embedding_model.predict(generator.flow(nodes))
        embeddings = pd.DataFrame(emb.squeeze(0), index=nodes)
        return embeddings

    def node_embedding(self, directory, directed=True, export_each_graph=False):

        file_pairs = prepare_samples(self.param)
        df_emb = pd.DataFrame(columns=range(self.param.DIMENSIONS))
        df_label = pd.DataFrame(columns=["label"])

        for file_pair in tqdm(file_pairs, desc="processing file pairs:"):
            file_graph = file_pair[0]
            file_pred = file_pair[1]
            label = file_pair[2]
            graph, node_pair_to_predict = generate_atom_graph(file_dir=directory,
                                                              file_graph=file_graph,
                                                              file_pred=file_pred,
                                                              directed=directed,
                                                              export=export_each_graph
                                                              )
            # skip the empty graphs
            if len(graph.nodes) == 0:
                continue
            stellargraph, vec_onehot = self.create_stellargraph(graph)
            embeddings = self.gcn_learn(stellargraph, vec_onehot)
            # it can happen that to be predicted nodes not exist in this atom graph
            try:
                emb_target_nodes = embeddings.loc[node_pair_to_predict, :]
                emb_serie = operate(emb_target_nodes, self.pattern)
            except:
                emb_serie = pd.Series(
                    index=range(self.param.DIMENSIONS),
                    dtype=np.float64
                )
            final_name = process_name(node_pair_to_predict)
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