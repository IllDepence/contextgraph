import warnings
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from stellargraph import StellarGraph
from stellargraph.data import UnsupervisedSampler
from stellargraph.layer import Attri2Vec, link_classification
from stellargraph.mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator
from graphml.preprocessor.graph_processing import prepare_samples, \
    process_name, operate, generate_atom_graph


class Attri2VecEmbedder():

    def __init__(self, param_object):
        '''
        Class that is used to create a Attri2Vec "Embedder", which
        can be used for creating node embedding using Atrri2Vec by calling
        node_embedding method

        Attributes
        ----------
        param_object: an object that stores all necessary hyperparameters
        pattern: the pattern of how the final embedding of two embeddings
                of targets nodes are calculated
        embeddings: used to store the embedding dataframe, not needed once
                    the embedding is stored as file

        Methods
        ----------
        create_stellargraph: takes single nx graph as input and create a
                StellarGraph. The StellarGraph needs to have numerical
                features for performing Attri2Vec. Here the features are
                set as onhot-encoded vectors of node types. Refer to
                https://stellargraph.readthedocs.io/en/stable/demos/node-classification/attri2vec-node-classification.html
        attri2vec_learn: perform attri2vec using created StellarGraph.
                Output is the processed embedding vector of target nodes
                in this graph.
        node_embedding: takes in the path of graph samples as parameter,
                calls the above two methods and generates as many embedding
                vectors as the number of "file_pairs". Number is defined
                in ../parameters.py. Returns embedding vectors in
                dataframe
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
        vec_onehot = enc.fit_transform(df).toarray()
        df_onehot = pd.DataFrame(vec_onehot,
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
            # node_type_attr="type",
            edge_type_attr="type",
            node_type_default="task"
        )
        return stellargraph


    def atrri2vec_learn(self, stellargraph):
        tf.config.experimental.enable_tensor_float_32_execution(False)
        nodes = list(stellargraph.nodes())
        unsupervised_samples = UnsupervisedSampler(
            stellargraph,
            nodes=nodes,
            length=self.param.WALK_LENGTH,
            number_of_walks=self.param.NUM_WALKS
        )
        generator = Attri2VecLinkGenerator(
            stellargraph,
            self.param.BATCH_SIZE
        )
        attri2vec = Attri2Vec(
            layer_sizes=[self.param.DIMENSIONS],
            generator=generator,
            bias=False,
            normalize=None
        )
        x_inp, x_out = attri2vec.in_out_tensors()
        prediction = link_classification(
            output_dim=1,
            output_act="sigmoid",
            edge_embedding_method="ip")(x_out)
        model = keras.Model(inputs=x_inp, outputs=prediction)
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.param.LEARNING_RATE
            ),
            loss=keras.losses.binary_crossentropy,
            metrics=[keras.metrics.binary_accuracy]
        )

        train_gen = generator.flow(unsupervised_samples)
        _ = model.fit(train_gen,
                      epochs=self.param.EPOCHS,
                      verbose=0,
                      use_multiprocessing=False,
                      workers=1,
                      shuffle=True
                      )
        x_inp_src, x_out_src = x_inp[0], x_out[0]
        embedding_model = keras.Model(
            inputs=x_inp_src,
            outputs=x_out_src
        )
        node_gen = Attri2VecNodeGenerator(
            stellargraph,
            self.param.BATCH_SIZE
        ).flow(nodes)
        node_embeddings = embedding_model.predict(
            node_gen,
            workers=self.param.WORKERS,
            verbose=0
        )
        embeddings = pd.DataFrame(node_embeddings, index=nodes)
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
            stellargraph = self.create_stellargraph(graph)
            embeddings = self.atrri2vec_learn(stellargraph)
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
