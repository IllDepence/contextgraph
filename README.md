# Data

### unarXive

* papers: `/home/ls3data/datasets/unarXive/papers/<id>.txt`
* refs DB: `/home/ls3data/datasets/unarXive/papers/refs.db`

### paperswithcode

* `/home/ls3data/datasets/paperswithcode`

# Code

* **exploration** `exploration/`
    * `notebooks/` - for *temporary* development and quick overviews. always move code to be re-used/shared into proper Python scripts
    * `pwc_matching_poc.py` - match entities from Papers With Code in unarXive paper plaintexts
    * `matching_result_ui/` - web UI to show matching results

* **preprocessing** `contextgraph/preprocessing/`
    * `crawler.py`
    * `dsets.py`  (requires `crawl_dataset_papers.py` output)
    * `meths.py`
    * `evaltbls.py`  (requires `preprocess_datasets.py` and `preprocess_methods.py` output)
    * `pprs.py`  (requires `preprocess_evaltables.py` output)
    * `cit_netw.py`  (requires `preprocess_papers.py` output)
    * `ppr_cntxts.py` (requires output of all of the above)
        * requires module `regex` (not `re`)
        * example use on icarus: `$ python3 add_paper_contexts.py --pwc_dir /home/ls3data/datasets/paperswithcode/preprocessed/ --unarxive_dir /opt/unarXive/unarXive-2020/papers/`

* **visualization** `contextgraph/visualization/`
    * ...

* **prediction** `contextgraph/prediction/`
    * ...


# TODOs

* preprocess paperswithcode to
    * jsonl file per given entity type (method, dataset, task, paper) ✔
    * extra entities (model, method-collection, collection-area) ✔
    * mappings using pwc URL slugs between entities ✔
* extend data with unarXive
    * citation graph ✔
    * extracted contexts ✔
        * more sophisticated context span determination (see `add_paper_contexts.py:204`)
* work with graph
    * PoC for working with a graph library ✔
    * add currently missing data
        * method collections and areas ✔
        * entity contexts ✔
        * topological encoding of paper publication order
    * create visualization
    * explore different settings (directed, undirected, additional edges for symmetrical relationships, special transitive relationships, etc.)
    * create function for generating train/val/test data
        * pairs of entities which
            * have at least one common paper
            * are of dissimilar type
            * have a using paper with a given minimum age (e.g. from 2010)
        * pruning of
            * all papers older that the first common paper
            * used_together edges
            * ... (more?)
* prediction
    * create a first set of data to evaluate and develop methods
        * prediction edge considerations:
            * “virtual” edges between diff. typ. entities w/ a common paper
            * edges between meths and dsets which have an evaltbl row
            * the union of both of the above
            * TODO: calculate numbers for the above
    * try simple methods
        * topology only
        * topology + pre-computed embeddings for node features
    * try GNNs
        * directed vs. undirected graph
        * typed vs. untyped edges
