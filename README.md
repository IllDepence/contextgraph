# Overview

* [Coordination](#coordination)
* [Usage](#usage)
* [Data used](development.md#data)
* [Development notes](development.md#directory-structure-overview)
* [Progress tracking](progress_tracking.md)

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
<summary>torch geometric</summary>

* node features
    * tasks
        * id (int)
        * type (one-hot encoding)
        * description (bag of words)
        * categories (one-hot enconding)
    * method
        * id (int)
        * type (one-hot encoding)
        * description (bag of words)
        * introduced\_year (int)
        * num\_papers (int)
    * dataset
        * id (int)
        * type (one-hot encoding)
        * description (bag of words)
        * introduced\_date (int)
        * modalities (one-hot encoding)
        * num\_papers (int)
        * data\_loaders (int (number of data loaders))

</details>
