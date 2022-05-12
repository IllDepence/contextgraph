# Data

### unarXive

* papers: `/home/ls3data/datasets/unarXive/papers/<id>.txt`
* refs DB: `/home/ls3data/datasets/unarXive/papers/refs.db`

### paperswithcode

* `/home/ls3data/datasets/paperswithcode`

# Code

* `paperswithcode_preprocessing/`
    * `crawl_dataset_papers.py`
    * `preprocess_datasets.py`  (requires `crawl_dataset_papers.py` output)
    * `preprocess_methods.py`
    * `preprocess_evaltables.py`  (requires `preprocess_datasets.py` and `preprocess_methods.py` output)
    * `preprocess_papers.py`  (requires `preprocess_evaltables.py` output)
    * `add_citation_network.py`  (requires `preprocess_papers.py` output)

* `paperswithcode_X_unarXive/`
    * `notebooks/` - for *temporary* development and quick overviews. always move code to be re-used/shared into proper Python scripts
    * `pwc_matching_poc.py` - match entities from Papers With Code in unarXive paper plaintexts
    * `matching_result_ui/` - web UI to show matching results

# TODOs

* preprocess paperswithcode to
    * jsonl file per given entity type (method, dataset, task, paper)
    * extra entities (model, method-collection, collection-area)
    * mappings using pwc URL slugs between entities
* extend data with unarXive
    * citation graph ✔
    * extracted contexts
