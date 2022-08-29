import os
import pandas as pd
from sklearn.model_selection import train_test_split
from contextgraph import config as cg_config
from bow_embedding import create_bow_embedding_for_ml
from node2vec_embedding import Node2VecEmbedder
from attri2vec_embedding import Attri2VecEmbedder
from gcn_embedding import GCNEmbedder

def create_embedding(path_to_store, param, embedding_method):

    if embedding_method=="node2vec":
        embedder = Node2VecEmbedder(param_object=param)
    elif embedding_method=="attri2vec":
        embedder = Attri2VecEmbedder(param_object=param)
    elif embedding_method=="gcn":
        embedder = GCNEmbedder(param_object=param)

    embeddings = embedder.node_embedding(
        directory=cg_config.graph_samples,
        directed=True
    )
    embeddings.to_csv(path_to_store)
    return embeddings


def preprocessing(dataframe):
    dataframe.loc[:, "label"] = dataframe.loc[:, "label"].map({'pos': 1, 'neg': 0})
    dataframe_cleaned = dataframe.dropna()
    return dataframe_cleaned


def get_training_data(data_path_node2vec, data_path_bow,
                      param, process_pattern_bow="avg",
                      embedding_method="node2vec", with_text_info=False
                      ):
    if os.path.exists(data_path_node2vec):
        embeddings_node2vec = pd.read_csv(data_path_node2vec, index_col=0)
    else:
        embeddings_node2vec = create_embedding(path_to_store=data_path_node2vec,
                                               param=param,
                                               embedding_method=embedding_method
                                               )
    embeddings = preprocessing(embeddings_node2vec)

    if with_text_info:
        if os.path.exists(data_path_bow):
            embeddings_bow = pd.read_csv(data_path_bow, index_col=0)
        else:
            embeddings_bow = create_bow_embedding_for_ml(
                embeddings,
                param,
                pattern=process_pattern_bow)
        embeddings_bow.to_csv(data_path_bow)
        embeddings = pd.concat([embeddings, embeddings_bow], axis=1)
        embeddings = embeddings.dropna()
    return embeddings


def split_data(dataframe, param, test_size=0.2):
    X = pd.concat(
        [dataframe.iloc[:, :param.DIMENSIONS], dataframe.iloc[:, param.DIMENSIONS+1:]],
        axis=1
    )
    y = dataframe.iloc[:, param.DIMENSIONS]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42
    )
    return X_train, X_test, y_train, y_test