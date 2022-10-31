import os
import re
import json
from tqdm import tqdm
import pandas as pd
from contextgraph import config as cg_config



def get_descriptions(file_pairs):
    nodes = []
    node_pairs = []
    for pair in tqdm(file_pairs, desc="getting description:"):
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


def clean_texts(text):
    text = text.lower()
    text_markup = re.sub('<.*?>', '', text)
    text_mentions = re.sub("@\S+", "", text_markup)
    text_market_tickers = re.sub("\$", "", text_mentions)
    text_url = re.sub(r'http\S+', '', text_market_tickers)
    text_hashtags = re.sub("#", "", text_url)
    text_punctuation = re.sub("[^-9A-Za-z ]", "", text_hashtags)
    return text_punctuation


def clean_dataframe(df):
    df = df.dropna()
    df = df.loc[df["description"].apply(lambda x: len(x)) > 0]
    df["description"] = df["description"].apply(clean_texts)
    return df