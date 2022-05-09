# Data

### unarXive

* papers: `/home/ls3data/datasets/unarXive/papers/<id>.txt`
* refs DB: `/home/ls3data/datasets/unarXive/papers/refs.db`

### paperswithcode

* `/home/ls3data/datasets/paperswithcode`

# Code

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
    * extracted contexts
    * citation graph
