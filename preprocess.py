import os
import contextgraph.config as cg_config
from contextgraph.preprocessing.crawler import crawl_dataset_papers
from contextgraph.preprocessing.util import ensure_graph_data_dir
from contextgraph.preprocessing.dsets import preprocess_datasets
from contextgraph.preprocessing.meths import preprocess_methods
from contextgraph.preprocessing.evaltbls import preprocess_evaltables
from contextgraph.preprocessing.pprs import preprocess_papers
from contextgraph.preprocessing.cit_netw import add_citation_network
from contextgraph.preprocessing.ppr_cntxts import add_paper_contexts


# check if dataset to paper information is already cralwed
# and if not, run crawler
ext_dsets_fp = os.path.join(
    cg_config.pwc_data_dir,
    cg_config.pwc_dsets_ext_fn
)
if not os.path.isfile(ext_dsets_fp):
    crawl_dataset_papers()

ensure_graph_data_dir()
print('preprocessing data sets')
preprocess_datasets()
print('preprocessing methods')
preprocess_methods()
print('preprocessing evaluation tables')
preprocess_evaltables()
print('preprocessing papers')
preprocess_papers()
print('adding citation network')
add_citation_network()
print('adding paper contexts')
add_paper_contexts()
