from contextgraph import config as cg_config
from parameters import Parameters
from node2vec_embedder import Node2VecEmbedder

param = Parameters()
node2vec_ = Node2VecEmbedder(param_object=param, pattern="avg")
embeddings = node2vec_.node_embedding(directory=cg_config.graph_samples,
                                      directed=True
                                     )
embeddings.to_pickle("/local/users/ujvxd/env/test2.pkl")