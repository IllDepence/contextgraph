import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
from stellargraph import StellarGraph
from stellargraph.layer import GAT
from stellargraph.mapper import FullBatchNodeGenerator
from graphml.preprocessor.graph_processing import prepare_samples, \
    process_name, operate, generate_atom_graph


class GATEmbedder():

    def __init__(self, param_object):
        '''
        Structure is highly alike with ../gcn_embedding.py,
        please refer to the documentation of GCNEmbedder and
        the example from official website for GAT embedding:
        https://stellargraph.readthedocs.io/en/stable/demos/node-classification/rgcn-node-classification.html
        '''

        self.param = param_object
        self.pattern = self.param.PATTERN_GRAPH
        self.embeddings = None

    def create_stellargraph(self, graph):
        df = pd.DataFrame(columns=["node", "type"])
        df["node"] = [node[0] for node in graph.nodes(data=True)]
        # TODO: deal with potential missing type,
        #  right now simply set as "task"
        df["type"] = [node[-1].get("type", "task")
                      for node in graph.nodes(data=True)]
        df = df.set_index("node")
        enc = OneHotEncoder(handle_unknown='ignore')
        df_onehot = pd.DataFrame(
            enc.fit_transform(df).toarray(),
            index=df.index,
            columns=enc.categories_
        )
        features_dict = dict()
        for row in df_onehot.iterrows():
            features_dict[row[0]] = list(row[1])
        nx.set_node_attributes(graph, features_dict, "feature")
        stellargraph = StellarGraph.from_networkx(
            graph,
            node_features="feature",
            edge_type_attr="type",
            node_type_default="task"
        )
        return stellargraph, df_onehot

    def gat_learn(self, stellargraph, df_onehot):

        train_targets, test_targets = model_selection.train_test_split(
            df_onehot,
            test_size=0.2
        )
        generator = FullBatchNodeGenerator(
            stellargraph,
            method="gat"
        )
        train_gen = generator.flow(
            train_targets.index,
            train_targets.to_numpy()
        )
        val_gen = generator.flow(
            test_targets.index,
            test_targets.to_numpy()
        )
        all_gen = generator.flow(df_onehot.index)

        gat = GAT(
            layer_sizes=[8, df_onehot.shape[1]],
            activations=["elu", "softmax"],
            attn_heads=int(self.param.DIMENSIONS/8),
            generator=generator,
            in_dropout=self.param.DROPOUT,
            attn_dropout=self.param.DROPOUT,
            normalize=None
        )
        x_inp, predictions = gat.in_out_tensors()
        model = tf.keras.Model(inputs=x_inp, outputs=predictions)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.param.LEARNING_RATE_GAT
            ),
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=["acc"]
        )
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.param.ES_PATIENCE,
            restore_best_weights=True
        )
        mc_callback = tf.keras.callbacks.ModelCheckpoint(
            "logs/best_model.h5",
            monitor="val_acc",
            save_best_only=True,
            save_weights_only=True
        )
        _ = model.fit(
            train_gen,
            epochs=self.param.EPOCHS_GAT,
            validation_data=val_gen,
            verbose=0,
            shuffle=False,
            callbacks=[es_callback, mc_callback]
        )
        emb_layer = next(l for l in model.layers
                         if l.name.startswith("graph_attention")
                         )
        embedding_model = tf.keras.Model(
            inputs=x_inp,
            outputs=emb_layer.output
        )
        emb = embedding_model.predict(all_gen)
        embeddings = pd.DataFrame(
            emb.squeeze(0),
            index=df_onehot.index
        )
        return embeddings

    def node_embedding(self,
                       directory,
                       directed=True,
                       export_each_graph=False
                       ):

        file_pairs = prepare_samples(self.param)
        df_emb = pd.DataFrame(columns=range(self.param.DIMENSIONS))
        df_label = pd.DataFrame(columns=["label"])

        for file_pair in tqdm(file_pairs, desc="processing file pairs:"):
            file_graph = file_pair[0]
            file_pred = file_pair[1]
            label = file_pair[2]
            graph, node_pair_to_predict = generate_atom_graph(
                file_dir=directory,
                file_graph=file_graph,
                file_pred=file_pred,
                directed=directed,
                export=export_each_graph
            )
            # skip the empty graphs
            if len(graph.nodes) == 0:
                continue
            stellargraph, df_onehot = self.create_stellargraph(graph)
            embeddings = self.gat_learn(stellargraph, df_onehot)
            # it can happen that to be predicted nodes
            # not exist in this atom graph
            try:
                emb_target_nodes = embeddings.loc[node_pair_to_predict, :]
                emb_serie = operate(emb_target_nodes, self.pattern)
            except Exception:
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
