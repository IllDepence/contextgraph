from node2vec import Node2Vec
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from data_processing import prepare_samples, process_name, operate, generate_atom_graph

class Node2VecEmbedder():

    def __init__(self, param_object):
        '''
        param_object: a object from dataclass that stores all necessary hyperparameter for Node2Vec
        pattern: the pattern of how the final embedding of two embeddings of targets nodes are calculated
        '''

        self.param = param_object
        self.pattern = self.param.PATTERN
        self.embeddings = None

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
