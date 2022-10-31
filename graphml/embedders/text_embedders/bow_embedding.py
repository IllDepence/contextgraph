import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from graphml.preprocessor.graph_processing import prepare_samples, process_name, operate
from graphml.preprocessor.text_processing import get_descriptions, clean_dataframe


def _get_bow_vectors(df):
    vectorizer = CountVectorizer(stop_words="english")
    result = vectorizer.fit_transform(df.description)
    df_bow = pd.DataFrame(
        result.toarray(),
        index=df.index,
        columns=vectorizer.get_feature_names_out()
    )
    df_bow = df_bow[~df_bow.index.duplicated(keep='first')]
    return df_bow


def create_bow_embedding_for_ml(df_graph_embedding, param):
    file_pairs = prepare_samples(param)
    df, node_pairs = get_descriptions(file_pairs)
    df = clean_dataframe(df)
    df_bow = _get_bow_vectors(df)
    df_emb = pd.DataFrame(columns=df_bow.columns)
    for node_pair in tqdm(node_pairs, desc="bow embedding for nodes:"):
        name_row = process_name(node_pair)
        if name_row in df_graph_embedding.index:
            try:
                df_2r = df_bow.loc[node_pair, :]
                emb_serie = operate(df_2r, pattern=param.PATTERN_TEXT)
            except:
                emb_serie = pd.Series(index=range(df_bow.shape[1]), dtype=np.float64)
            df_emb.loc[name_row, :] = emb_serie
    return df_emb
