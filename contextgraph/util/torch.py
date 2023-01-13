""" Make graph data usable with PyTorch Geometric
"""

import os
import torch
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Dataset
from contextgraph.config import graph_data_dir
from contextgraph.util.graph import _load_node_tuples,\
                                    _load_entity_combi_edge_tuples


class EntityCombiGraph(Dataset):
    def __init__(
        self, root=None, transform=None, pre_transform=None, pre_filter=None
    ):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['']

    @property
    def processed_file_names(self):
        return ['data_0.pt']

    @property
    def processed_dir(self) -> str:
        return os.path.join(graph_data_dir, 'pyg_data')

    def process(self):
        """ Note: usually iterates over all samples.
            For testing we just load the final graph as a
            single sample
        """

        node_tuples = _load_node_tuples(entities_only=True)
        edge_tuples = _load_entity_combi_edge_tuples(
            final_node_set=set([ntup[0] for ntup in node_tuples]),
            scheme='weight'
        )
        G = nx.Graph()
        node_tuples_numerical = []
        for ntup in node_tuples:
            # features
            # - already numeric
            #   - method: introduced_year, num_papers
            #   - dataset: year, month, day, num_papers
            #   - task: n/a
            attribs_num = {
                'num_papers': ntup[1].get('num_papers', -1)
            }
            node_tuples_numerical.append(
                (ntup[0], attribs_num)
            )
        G.add_nodes_from(node_tuples_numerical)
        G.add_edges_from(edge_tuples)
        data = from_networkx(
            G,
            group_node_attrs=['num_papers'],
            group_edge_attrs=['weight']
        )
        idx = 0
        torch.save(
            data,
            os.path.join(self.processed_dir, f'data_{idx}.pt')
        )

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data
