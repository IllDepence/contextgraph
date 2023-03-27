# CoCon

A data set capturing the combined use of research artifacts, contextualized through academic publication text.

* [Data on Zenodo](https://doi.org/10.5281/zenodo.7774292)
* [Code](#usage) (`contexthraph/`)

## Usage

### NetworkX

The methods described below can be imported from `contextgraph.util.graph`.  
You’ll also want to `import networkx as nx` in your code.

##### load\_full\_graph()

Returns the full graph in which all nodes and edges types.

parameter | values | default | explanation
--------- | ------ | ------- | -----------
shallow   | bool   | False   | True: all node and edge features except for type will be discarded.
&zwnj;    | &zwnj; | &zwnj;  | False: nodes and edges have features according to the data available on Papers with Code.
directed  | bool   | True    | True: Return the graph as an nx.DiGraph (for an overview of directions of edges see `_load_edge_tuples` in `contextgraph/util/graph.py`).
&zwnj;    | &zwnj; | &zwnj;  | False: Return the graph as an nx.Graph.
with\_contexts  | bool | False | True: Also load “context” nodes (each appearance of a used entity in a paper results in a context node connected as entity--part\_of-&gt;context--part\_of-&gt;paper). **Be aware** that this will result in *a lot* of additional nodes.
&zwnj;    | &zwnj; | &zwnj;  | False: Don’t load context nodes.

##### load\_entity\_combi\_graph()

Load graph only containing the entity nodes connected by edges which represent their co-occurrence papers based on a *scheme* setting described below. (**NOTE**: to access edge attributes later, G.edges(data=True) has to be used!)

parameter | values | default | explanation
--------- | ------ | ------- | -----------
scheme    | {'weight', 'sequence'} | 'sequence' | 'sequence': edges have two properties (1) `linker_sequence` (a list of the co-occurrence paper IDs) and (2) `interaction_sequence` (a list of integers to be understood as “time steps”. Each integer value is the `<month>` in which a co-occurrence paper is published. The month in which the very first co-occurrence paper within the returned graph is published in month 0).
&zwnj;    | &zwnj; | &zwnj;  | 'weight': edges have a weight attribute that denotes the number of co-occurrence papers that exist between the two entities.

<details>
<summary>graph schema</summary>

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
* edge features
    * [scheme](#load_entity_combi_graph) = `weight`
        * “weight”
    * [scheme](#load_entity_combi_graph) = `sequence`
        * “interaction\_sequence”
        * “linker\_sequence”

</details>

### PyTorch Geometric

The methods described below can be imported from `contextgraph.util.torch`.

##### load\_entity\_combi\_graph()

Load graph only containing the entity nodes connected by edges which represent their co-occurrence papers. Edge weights represent the number of co-occurrence papers.

<details>
<summary>graph schema</summary>

* node features
    * id (ordinal) (0..\<num\_nodes\>)
    * type (ordinal) (dataset: 0, method: 1, model: 2, task: 3)
    * description (transformer based embedding)
* edge features
    * “weight” (=number of combined use papers, see [load\_entity\_combi\_graph() scheme parameter](#load_entity_combi_graph))

</details>


## Preprocessing

* Set paths in `contexthraph/config.py`
* Run `$ python3 preprocess.py`
* Run `$ python3 precomp_descr_embs.py`
