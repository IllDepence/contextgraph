# Code

* **prediction** `contextgraph/prediction/`
    * *training sample generation*
        * `export_cytoscape_data.py` &gt; `export_samples_cyto()`
    * sampling of negative training examples
        * “currupt edge” (one entity of co-occurrence edge swapped to random)
            * pruning problem: no common using papers through which to determine paper publication threshold
            * → group cooc edges by year of first cooc ppr
            * → from each year set take pairs of cooc edges with disjoint cooc ppr sets
            * → create negative examples by switching out entity nodes between such pairs
        * take pruned 2-neighborhood graphs of positive examples and select other random pairs of enitites (consider for training. *maybe not suited for evaluation*)
            * appropriate (also well suited?) for training
            * *not* realistic for testing
                * **TODO**: create test set according to realistic application setting of model
                * number of possible random entity combinations might be too dominating  
                  → consider restriction by area (only NLP methods w/ NLP tasks or similar)
    * alternative prediction task
        * in: input entity pair (mby w/ focus on cited papers of using papers)
        * out: set of papers that the co-occurrence papers cite
        * intuition: if you were to write a combining paper, which papers would you need to cite/would you be influenced by

* **visualization** `contextgraph/visualization/`
    * `show_sample.py` (using nx + matplotlip)
    * `export_cytoscape_data.py` + Cytoscape (for manual inspection)
        * side note: Cytoscape has a REST API: `./cytoscape.sh -R 8888` ([doku](https://manual.cytoscape.org/en/3.5.0/Programmatic_Access_to_Cytoscape_Features_Scripting.html))

# **stats**
    * TODO: general status about Graph and involved entities
        * e.g. distribution of entity use numbers/frequency

* **preprocessing**
    * `preprocess.py` (uses paths in `contextgraph/config.py`)
    *  `contextgraph/preprocessing/`
        * `crawler.py`
        * `dsets.py`  (requires `crawl_dataset_papers.py` output)
        * `meths.py`
        * `evaltbls.py`  (requires `preprocess_datasets.py` and `preprocess_methods.py` output)
        * `pprs.py`  (requires `preprocess_evaltables.py` output)
        * `cit_netw.py`  (requires `preprocess_papers.py` output)
        * `ppr_cntxts.py` (requires output of all of the above)
            * requires module `regex` (not `re`)

* **exploration** `exploration/`
    * `notebooks/` - for *temporary* development and quick overviews. always move code to be re-used/shared into proper Python scripts
    * `pwc_matching_poc.py` - match entities from Papers With Code in unarXive paper plaintexts
    * `matching_result_ui/` - web UI to show matching results


# Conceptual considerations

### Modeling for basic GNN

* maybe get rid over ppr nodes and create `used_rogether` edges


# Data

### unarXive

* papers: `/home/ls3data/datasets/unarXive/papers/<id>.txt`
* refs DB: `/home/ls3data/datasets/unarXive/papers/refs.db`

### paperswithcode

* `/home/ls3data/datasets/paperswithcode`


# TODOs

* preprocess paperswithcode to ✔
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
            * are of dissimilar type (also not model+method)
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
