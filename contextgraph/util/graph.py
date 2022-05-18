import csv
import os
import json
import networkx as nx
from contextgraph import config


def _load_node_tuples(data_dir):
    """ loads nodes as (<id>, <properties>) tuples
    """

    pprs_fn = 'papers.jsonl'
    meths_fn = 'methods.jsonl'
    dsets_fn = 'datasets.jsonl'
    tasks_fn = 'tasks.jsonl'
    modls_fn = 'models.jsonl'
    entity_tuples = []
    for fn in [pprs_fn, meths_fn, dsets_fn, tasks_fn, modls_fn]:
        with open(os.path.join(data_dir, fn)) as f:
            for line in f:
                entity = json.loads(line)
                entity_tuples.append((entity['id'], entity))
    return entity_tuples


def _load_edge_tuples(data_dir):
    """ loads edges as (<id>, <properties>) tuples
    """

    # used_in_paper
    meths_to_pprs_fn = 'methods_to_papers.csv'
    dsets_to_pprs_fn = 'datasets_to_papers.csv'
    tasks_to_pprs_fn = 'tasks_to_papers.csv'
    modls_to_pprs_fn = 'models_to_papers.csv'
    # used_together (tread as symmetrical?)
    meths_to_dsets_fn = 'methods_to_datasets.csv'
    dsets_to_tasks_fn = 'datasets_to_tasks.csv'
    # part_of
    meths_to_colls_fn = 'methods_to_collections.csv'
    tasks_to_subtasks_fn = 'tasks_to_subtasks.csv'
    # cites
    pprs_to_pprs_fn = 'papers_to_papers.csv'
    edge_types = [
        [meths_to_pprs_fn, 'used_in_paper'],
        [dsets_to_pprs_fn, 'used_in_paper'],
        [tasks_to_pprs_fn, 'used_in_paper'],
        [modls_to_pprs_fn, 'used_in_paper'],
        [meths_to_dsets_fn, 'used_together'],
        [dsets_to_tasks_fn, 'used_together'],
        [meths_to_colls_fn, 'part_of'],
        [tasks_to_subtasks_fn, 'part_of'],
        [pprs_to_pprs_fn, 'cites']
    ]
    edges = []
    for (fn, edge_type) in edge_types:
        header_idxs = [0, 1]
        if edge_type == 'cites':
            header_idxs = [3, 4]
        with open(os.path.join(data_dir, fn)) as f:
            csv_reader = csv.DictReader(
                f,
                delimiter=',',
                quoting=csv.QUOTE_NONE
            )
            headers = csv_reader.fieldnames
            for row in csv_reader:
                tail_id = row[headers[header_idxs[0]]]
                head_id = row[headers[header_idxs[1]]]
                edges.append(
                    (
                        tail_id,
                        head_id,
                        {'type': edge_type}
                    )
                )
    return edges


def load_graph():
    data_dir = config.graph_data_dir
    node_tuples = _load_node_tuples(data_dir)
    edge_tuples = _load_edge_tuples(data_dir)
    G = nx.DiGraph()
    G.add_nodes_from(node_tuples)
    G.add_edges_from(edge_tuples)
    return G
