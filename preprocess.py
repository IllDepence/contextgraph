import os
import contextgraph.config as cg_config
import contextgraph.preprocessing as cg_prep

# check if dataset to paper information is already cralwed
# and if not, run crawler
ext_dsets_fp = os.path.join(
    cg_config.pwc_data_dir,
    cg_config.pwc_datasets_ext_fn
)
if not os.path.isfile(ext_dsets_fp):
    cg_prep.crawler.crawl_dataset_papers()

# preprocess data sets
cg_prep.preprocess_datasets.preprocess_datasets()
# preprocess methods
cg_prep.preprocess_methods.preprocess_methods()
# preprocess evaluation tables
cg_prep.preprocess_evaltables.preprocess_evaltables()
# preprocess papers
cg_prep.preprocess_papers.preprocess_papers()
# add citation network
cg_prep.add_citation_network.add_citation_network()
