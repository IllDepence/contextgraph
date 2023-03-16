# Overview

* [Coordination](#coordination)
* [Usage](#usage)
* [Development](#development)

# Coordination

### Issues

* Create issues to track tasks and point out problems.

### Chat

* Use the project’s **[Matrix channel](https://matrix.to/#/!TJpokbSjPiQzYMsiFC:kit.edu?via=kit.edu)** to communication with everyone involved.

# Usage

### Loading the graph data

The methods described below can be impored from `contextgraph.util.graph`.  
You’ll also want to `import networkx as nx` in your code.

##### load\_full\_graph()

Returns a version of the graph in which nodes and endges have properties.

parameter | values | default | explanation
--------- | ------ | ------- | -----------
shallow   | bool   | False   | True: all node and edge features except for type will be discarded.
&zwnj;    | &zwnj; | &zwnj;  | False: nodes and edges have features according to the data available on Papers with Code.
directed  | bool   | True    | True: Return the graph as an nx.DiGraph.
&zwnj;    | &zwnj; | &zwnj;  | False: Return the graph as an nx.Graph.
directed  | bool   | True    | True: Return the graph as an nx.DiGraph (for an overview of directions of edges see `_load_edge_tuples` in `contextgraph/util/graph.py`).
&zwnj;    | &zwnj; | &zwnj;  | False: Return the graph as an nx.Graph.
with\_contexts  | bool | False | True: Also load “context” nodes (each appearance of a used entity in a paper results in a context node connected as entity--part\_of-&gt;context--part\_of-&gt;paper). **Be aware** that this will result in *a lot* of additional nodes.
&zwnj;    | &zwnj; | &zwnj;  | False: Don’t load context nodes.

##### load\_entity\_combi\_graph()

Load graph only containing the entity nodes connected by edges which represent their co-occurrence papers based on a *scheme* setting described below. (**NOTE**: to access edge attributes later, G.edges(data=True) has to be used!)

parameter | values | default | explanation
--------- | ------ | ------- | -----------
scheme    | {'weight', 'sequence'} | 'sequence' | 'sequence': edges have two properties (1) `linker_sequence` (a list of the co-occurrence paper IDs) and (2) `interaction_sequence` (a list of integers to be understood as “time steps”. Each integer value is the `<month>` in which a co-occurrence paper is published. The month in which the very first co-occurrence paper within the returned graph is published in month 0).
&zwnj;    | &zwnj; | &zwnj;  | 'weight': edges have a weight attribute that denotes the number of co-occurrene papers that exist between the two entities.


# Graph schema

### Full graph

(what you get using `load_full_graph()`)

TODO

### Entitiy combi graph

(what you get using `load_entity_combi_graph()`)

<details>
<summary>networkx</summary>

* node features
    * tasks
        * id (str)
        * type (str)
        * name (str)
        * description (str)
        * categories (list)
    * method
        * url (str)
        * name (str)
        * full\_name (str)
        * description (str)
        * paper (dict)
        * introduced\_year (int)
        * source\_url (str)
        * source\_title (str)
        * code\_snippet_url (str)
        * num\_papers (int)
        * id (str)
        * type (str)
    * model
        * id (str)
        * type (str)
        * name (str)
        * using\_paper\_titles (list)
        * evaluations (list)
    * dataset
        * url (str)
        * name (str)
        * full\_name (str)
        * homepage (str)
        * description (str)
        * paper (dict)
        * introduced_date (str)
        * warning (NoneType)
        * modalities (list)
        * languages (list)
        * num_papers (int)
        * data\_loaders (list)
        * id (str)
        * type (str)
        * year (int)
        * month (int)
        * day (int)
        * variant\_surface\_forms (list)

</details>

<details>
<summary>torch geometric (current - 23/03/16)</summary>

* node features
    * id (ordinal) (0..\<num\_nodes\>)
    * type (ordinal) (dataset: 0, method: 1, model: 2, task: 3)
    * description (transformer based embedding)
* edge features
    * “weight” (=number of combined use papers, see [load\_entity\_combi\_graph() scheme parameter](#load_entity_combi_graph))

</details>

<details>
<summary>torch geometric (ideas/notes)</summary>

* node features
    * tasks
        * id (ordinal)
        * type (ordinal)
        * description (bag of words)
        * categories (one-hot enconding)
    * method
        * id (ordinal)
        * type (ordinal)
        * description (bag of words)
        * introduced\_year (int)
        * num\_papers (int)
    * model
        * id (ordinal)
        * type (ordinal)
        * num\_papers (int) (derived from using\_paper\_titles)
        * evaluations (int (number of evaluations))
    * dataset
        * id (ordinal)
        * type (ordinal)
        * description (bag of words)
        * introduced\_date (int)
        * modalities (one-hot encoding)
        * num\_papers (int)
        * data\_loaders (int (number of data loaders))

</details>

# Development

**Note:** all relevant code should be kept inside the `contextgraph` module.

Below an outline of how the code works. (**Last updated**: 23/03/16)

### Preprocessing

From raw unarXive and Papers with Code data to JSONL and CSV files. (Already done. Data saved in `ls3data/datasets/paperswithcode_2023/preprocessed/`)

* `preprocess.py` (uses paths in `contextgraph/config.py`) calls methods from `contextgraph/preprocessing/` in the following order
    * `crawler.py` crawls extra data from Papers with Code API that is needed but missing from their official data dumps on Github
    * `dsets.py`  (requires `crawl_dataset_papers.py` output)  preprocesses dataset and (preliminary) task data
    * `meths.py`  preprocesses method data
    * `evaltbls.py`  (requires `preprocess_datasets.py` and `preprocess_methods.py` output) preprocesses task and model data
    * `pprs.py`  (requires `preprocess_evaltables.py` output) preprocesses paper data
    * `cit_netw.py`  (requires `preprocess_papers.py` output) adds the unarXive citation network
    * `ppr_cntxts.py` (requires output of all of the above) adds artifact usage contexts from unarXive full-texts

### networkx graph loading

Methods in `contextgraph.util.graph`.

**module internal methods**

* `_load_node_tuples` loads node data as `(<id>, <properties>)` tuples. These tuples can directly be loaded with a networkx graph’s `add_nodes_from` method
    * if parameter `with_contexts` is True, artifact usage context nodes will also be loaded (False by default)
    * if parameter `entities_only` is True, only research artifact nodes will be loaded (False by default)
* `_load_edge_tuples` loads node data as `(<id>, <properties>)` tuples. These tuples can directly be loaded with a networkx graph’s `add_edges_from` method
    * `with_contexts` should be set to the same as wen calling `_load_node_tuples`
    * `final_node_set`, a set of node IDs, should be provided when loading an “entity combi graph” (graph with only artifact nodes and “combining paper edges”)

**methods exposed by the module**

See [usage documentation above](#usage).

### torch graph loading

Methods in `contextgraph.util.torch`.

**module internal methods**

* `_get_artifact_description` returns the description attribute of a node (or generates one in case of model nodes)
* `_embed_string_atrs_tfidf` returns tf/idf vectors for the full list of node description attributes within the graph

**methods exposed by the module**

* `load_full_graph`: not implemented yet
* `load_entity_combi_graph` return the entity combi graph in a form usable with torch geometric, and operates as follows
    * load node and edge tuples from the JSONL and CSV data
    * converts node and edge attributes to a numerical form
        * see [torch geometric schema](#entitiy-combi-graph) in the graph schema section above
        * **Note**: transformer based node description emebddings are currently pre-computed with the script `precomp_descr_embs.py` — this should be integrated into a method `_embed_string_atrs_transformer` within `contextgraph.util.torch` (to save time the method should persist embeddings once generated and re-load them when called later again)
    * creates a networkx graph from the converted node and edge tuples
    * uses `torch_geometric.utils.convert.from_networkx` to convert the graph
