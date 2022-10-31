import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from graphml.preprocessor.graph_processing import prepare_samples, process_name, operate
from graphml.preprocessor.text_processing import get_descriptions, clean_dataframe


def _get_embeddings(text, tokenizer, model, param):
    bert_tokenizer = tokenizer(text,
                               padding='max_length',
                               max_length=param.MAX_LENGTH,
                               truncation=True,
                               return_tensors='pt'
                               )
    indexed_tokens = bert_tokenizer['input_ids']
    segments_ids = bert_tokenizer['attention_mask']
    with torch.no_grad():
        outputs = model(indexed_tokens, segments_ids)
        # variable 'hidden_state' has 13 layer, each has torch.Size([1, 512, 768])
        hidden_states = outputs[2]
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1, 0, 2)

    token_vecs = []
    for token in token_embeddings:
        mean_last_4_layers = torch.mean(torch.stack((token[-1],
                                                     token[-2],
                                                     token[-3],
                                                     token[-4]),
                                                    dim=0),
                                        dim=0)
        token_vecs.append(mean_last_4_layers.numpy())
    embedding = np.mean(token_vecs, axis=0)
    return embedding

def create_bert_embedding_for_ml(df_graph_embedding, param, tokenizer, model):
    file_pairs = prepare_samples(param)
    df, node_pairs = get_descriptions(file_pairs)
    df = clean_dataframe(df)

    df_bert = pd.DataFrame(df["description"].apply(lambda x: _get_embeddings(x, tokenizer, model, param)))
    df_bert = pd.DataFrame(df_bert["description"].tolist(), index=df_bert.index)

    for node_pair in tqdm(node_pairs, desc="bert embedding for nodes:"):
        name_row = process_name(node_pair)
        if name_row in df_graph_embedding.index:
            try:
                df_2r = df_bert.loc[node_pair, :]
                emb_serie = operate(df_2r, pattern=param.PATTERN_TEXT)
            except:
                emb_serie = pd.Series(index=range(df_bert.shape[1]), dtype=np.float64)
            df_bert.loc[name_row, :] = emb_serie
    return df_bert