import os
import pandas as pd
from sklearn.model_selection import train_test_split
from contextgraph import config as cg_config
from parameters import Parameters
from node2vec_embedder import Node2VecEmbedder
from ml_models import create_models, cross_validate_models, get_performance_of_cv, evaluate


def create_embedding(path_to_store):
    param = Parameters()
    node2vec_ = Node2VecEmbedder(param_object=param, pattern="avg")
    embeddings = node2vec_.node_embedding(directory=cg_config.graph_samples,
                                          directed=True
                                         )
    embeddings.to_pickle(path_to_store)
    return embeddings


def get_training_data(path_to_file):
    if os.path.exists(path_to_file):
        embeddings = pd.read_csv(path_to_file, index_col=0)
    else:
        embeddings = create_embedding()
    return embeddings


def preprocessing(dataframe):
    dataframe.loc[:, "label"] = dataframe.loc[:, "label"].map({'pos': 1, 'neg': 0})
    dataframe_cleaned = dataframe.dropna()
    return dataframe_cleaned


def split_data(dataframe, test_size=0.2):
    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=42
                                                        )
    return X_train, X_test, y_train, y_test

def main(data_path):
    embeddings = get_training_data(path_to_file=data_path)
    embeddings = preprocessing(embeddings)
    X_train, X_test, y_train, y_test = split_data(embeddings, test_size=0.2)
    classifiers = create_models()
    results = cross_validate_models(model_dict=classifiers,
                                    training_data=X_train,
                                    training_label=y_train
                                    )
    performance_cv = get_performance_of_cv(results)
    evaluation = evaluate(results_cv=results,
                          test_data=X_test,
                          test_label=y_test,
                          display=True
                          )
    return performance_cv, evaluation

if __name__ == "__main__":
    path = "/local/users/ujvxd/env/embeddings_graph_sample.csv"
    performance_cv, evaluation = main(path)