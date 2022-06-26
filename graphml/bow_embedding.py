import re
import os
import json
import pandas as pd
import numpy as np
from contextgraph import config as cg_config
from sklearn.feature_extraction.text import CountVectorizer
from data_processing import prepare_samples, process_name, operate


def _get_descriptions(file_pairs):
    nodes = []
    node_pairs = []
    for pair in file_pairs:
        with open(os.path.join(cg_config.graph_samples, pair[1])) as f:
            info = json.load(f)
        node_pair = info.get("edge", [])
        node_pairs.append(node_pair)
        for node in node_pair:
            nodes.append(node)
    files = [
        cg_config.graph_dsets_fn,
        cg_config.graph_meths_fn,
        cg_config.graph_modls_fn,
        cg_config.graph_tasks_fn
    ]
    df = pd.DataFrame(index=nodes, columns=["description"])
    for file in files:
        with open(os.path.join(cg_config.graph_data_dir, file)) as f:
            for line in f:
                entity = json.loads(line)
                entity_id = entity.get("id", "")
                if entity_id in nodes:
                    description = entity.get("description", "")
                    df.loc[entity_id, "description"] = description
    return df, node_pairs


def _clean_texts(text):
    text = text.lower()
    text_markup = re.sub('<.*?>', '', text)
    text_mentions = re.sub("@\S+", "", text_markup)
    text_market_tickers = re.sub("\$", "", text_mentions)
    text_url = re.sub(r'http\S+', '', text_market_tickers)
    text_hashtags = re.sub("#", "", text_url)
    text_punctuation = re.sub("[^-9A-Za-z ]", "", text_hashtags)
    return text_punctuation


def _clean_dataframe(df):
    df = df.dropna()
    df = df.loc[df.description.apply(lambda x: len(x)) > 0]
    df["description"] = df["description"].apply(_clean_texts)
    return df


def _get_bow_vectors(df):
    vectorizer = CountVectorizer()
    result = vectorizer.fit_transform(df.description)
    df_bow = pd.DataFrame(
        result.toarray(),
        index=df.index,
        columns=vectorizer.get_feature_names_out()
    )
    df_bow = df_bow[~df_bow.index.duplicated(keep='first')]
    return df_bow


def create_bow_embedding_for_ml(df_node2vec, param, pattern="avg"):
    file_pairs = prepare_samples(param)
    df, node_pairs = _get_descriptions(file_pairs)
    df = _clean_dataframe(df)
    df_bow = _get_bow_vectors(df)
    df_emb = pd.DataFrame(columns=df_bow.columns)
    for node_pair in node_pairs:
        name_row = process_name(node_pair)
        if name_row in df_node2vec.index:
            try:
                df_2r = df_bow.loc[node_pair, :]
                emb_serie = operate(df_2r, pattern=pattern)
            except:
                emb_serie = pd.Series(index=range(df_bow.shape[1]), dtype=np.float64)
            df_emb.loc[name_row, :] = emb_serie
    return df_emb
