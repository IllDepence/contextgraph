import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from graphml.preprocessor.graph_processing import prepare_samples, \
    process_name, operate
from graphml.preprocessor.text_processing import get_descriptions, \
    clean_dataframe


def _get_bow_vectors(df):
    '''
    Internal use only. Create BOW vectors based all the text
    data that exists in "df"
    '''
    vectorizer = CountVectorizer(stop_words="english")
    result = vectorizer.fit_transform(df.description)
    df_bow = pd.DataFrame(
        result.toarray(),
        index=df.index,
        columns=vectorizer.get_feature_names_out()
    )
    df_bow = df_bow[~df_bow.index.duplicated(keep='first')]
    return df_bow


def create_bow_embedding_for_ml(
        df_graph_embedding,
        param
):
    '''
    Create the dataframe that contains the BOW embedding vectors,
    which ready to be concatenated with graph embedding vectors
    for performing ML activities

    Paramets
    ----------
    df_graph_embedding: embedding dataframe of graphs, used for
                BOW embedding results to find respective target
                nodes, so the two dataframe can be concatenated
                without error
    param: object of class Parameters
    '''
    file_pairs = prepare_samples(param)
    df, node_pairs = get_descriptions(file_pairs)
    df = clean_dataframe(df)
    df_bow = _get_bow_vectors(df)
    df_emb = pd.DataFrame(columns=df_bow.columns)
    for node_pair in tqdm(
            node_pairs,
            desc="bow embedding for nodes:"
    ):
        name_row = process_name(node_pair)
        if name_row in df_graph_embedding.index:
            try:
                df_2r = df_bow.loc[node_pair, :]
                emb_serie = operate(
                    df_2r,
                    pattern=param.PATTERN_TEXT
                )
            except:
                emb_serie = pd.Series(
                    index=range(df_bow.shape[1]),
                    dtype=np.float64
                )
            df_emb.loc[name_row, :] = emb_serie
    return df_emb
