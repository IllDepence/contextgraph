import os
import pandas as pd
from contextgraph import config as cg_config
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from graphml.embedders.text_embedders.bow_embedding import create_bow_embedding_for_ml
from graphml.embedders.text_embedders.bert_embedding import create_bert_embedding_for_ml
from graphml.embedders.graph_embedders.node2vec_embedding import Node2VecEmbedder
from graphml.embedders.graph_embedders.attri2vec_embedding import Attri2VecEmbedder
from graphml.embedders.graph_embedders.gcn_embedding import GCNEmbedder
from graphml.embedders.graph_embedders.rgcn_embedding import RGCNEmbedder
from graphml.embedders.graph_embedders.gat_embedding import GATEmbedder


def create_embedding(path_to_store, param, embedding_method):

    if embedding_method=="node2vec":
        embedder = Node2VecEmbedder(param_object=param)
    elif embedding_method=="attri2vec":
        embedder = Attri2VecEmbedder(param_object=param)
    elif embedding_method=="gcn":
        embedder = GCNEmbedder(param_object=param)
    elif embedding_method=="rgcn":
        embedder = RGCNEmbedder(param_object=param)
    elif embedding_method=="gat":
        embedder = GATEmbedder(param_object=param)
    else:
        raise Exception ("Unsupported graph embedding method is used, please check the input")
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

def get_training_data(data_path,
                      data_path_description,
                      param,
                      embedding_method="node2vec",
                      with_text_info=False,
                      text_embedding_method="bow"
                      ):
    if os.path.exists(data_path):
        embeddings = pd.read_csv(data_path, index_col=0)
    else:
        embeddings = create_embedding(path_to_store=data_path,
                                      param=param,
                                      embedding_method=embedding_method
                                      )
    embeddings = preprocessing(embeddings)

    if with_text_info:
        if os.path.exists(data_path_description):
            embeddings_text = pd.read_csv(data_path_description, index_col=0)
        else:
            if text_embedding_method == "bow":
                embeddings_text = create_bow_embedding_for_ml(
                    df_graph_embedding=embeddings,
                    param=param)
                embeddings_text.to_csv(data_path_description)
            elif text_embedding_method == "bert":
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
                _ = model.eval()
                embeddings_text = create_bert_embedding_for_ml(
                    df_graph_embedding=embeddings,
                    param=param,
                    tokenizer=tokenizer,
                    model=model)
                embeddings_text.to_csv(data_path_description)
                embeddings_text.columns = embeddings_text.columns.astype(str)

        if text_embedding_method == "bow" and embeddings_text.shape[0] > param.DIMENSIONS:
            pca = PCA(n_components=param.DIMENSIONS)
            pcs = pca.fit_transform(embeddings_text.dropna())
            embeddings_text = pd.DataFrame(pcs, index=embeddings_text.dropna().index)
            embeddings_text.columns = embeddings_text.columns.astype(str)

        embeddings = embeddings.merge(embeddings_text,
                                      how="inner",
                                      left_index=True,
                                      right_index=True
                                      ).dropna()
    return embeddings

def split_data(dataframe, param, realistic_testset=False, test_size=0.3):
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
    if realistic_testset:
        X_test, y_test = create_realistic_testset(X_test, y_test, param)
    return X_train, X_test, y_train, y_test

def create_realistic_testset(X_test, y_test, param):
    df_test = pd.concat([X_test, y_test], axis=1)
    num_pos, num_neg = y_test.value_counts().values
    pos_neg_ratio_data = num_pos / num_neg
    if pos_neg_ratio_data > param.POS_NEG_RATIO:
        sample_ratio = (num_neg * param.POS_NEG_RATIO) / num_pos
        sample_label = 1
    elif pos_neg_ratio_data < param.POS_NEG_RATIO:
        sample_ratio = num_pos / (num_pos*param.POS_NEG_RATIO)
        sample_label = 0
    else:
        return df_test
    df_test_pos = df_test[df_test.label == sample_label].sample(frac=sample_ratio,
                                                                replace=False,
                                                                random_state=42
                                                                )
    df_test = pd.concat([df_test_pos, df_test[df_test.label == (1-sample_label)]], axis=0)
    return df_test.iloc[:, :-1], df_test.iloc[:, -1]



