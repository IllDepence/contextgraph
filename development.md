# Data

### unarXive

* papers: `/home/ls3data/datasets/unarXive/papers/<id>.txt`
* refs DB: `/home/ls3data/datasets/unarXive/papers/refs.db`

### paperswithcode

* `/home/ls3data/datasets/paperswithcode`

# Directory structure overview

I.e., how the code is organized and what it is doing. Also some notes mixed within (← maybe should be moved to `progress_tracking.md`).

* **prediction** `contextgraph/prediction/`
    * *training sample generation*
        * `export_cytoscape_data.py` &gt; `export_samples_cyto()`
        * using two-hop neighborhoods took ~21h for the whole graph
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
    * *TODO*: general status about Graph and involved entities
        * e.g. distribution of entity use numbers/frequency
* **graph status**
    * On the whole graph level
        * The complete whole graph has 284760 nodes in total, over 91% of them are "paper"
        * The whole graph has an average degree of 8.1, the largest degree is 145232 for the entity "pwc:task/classification", around 98.7% percent of all entities have a degree lower than 50
        * Based on node type, a small portion of "task" has a degree over 1000; for "paper" and "task", the degrees of almost all entities are below 1000; degrees of "dataset" nodes are obviously below 1000; although there is a large amount of "model" type of nodes, their degrees are mostly below 3. (Details can be found in Notebook "statistical analysis")
    * On the samples level 
        * An average degree of 5 of graphs seem to be usual, a small portion of samples has relatively higher average degree, but no higher than 25
        * While the average degrees of sample graphs have no big difference across positive and negative classes, the average degree of target nodes is significantly higher for positive samples than negative ones, suggesting that positve samples are indeed more "well-connected" (more often combined with other entities)  

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
