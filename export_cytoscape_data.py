import os
import sys
import json
import networkx as nx
from contextgraph.util.graph import load_graph, get_pair_graphs


def export_cyto(fp, G):
    print(f'saving to {fp}')
    with open(fp, 'w') as f:
        f.write(
            json.dumps(nx.cytoscape_data(G))
        )


def export_samples_cyto(fp, num_samples):
    print('loading graph')
    G = load_graph(shallow=True)
    print('generating samples')
    samples = get_pair_graphs(num_samples, G)
    for i, sample in enumerate(samples):
        fp_pre, ext = os.path.splitext(fp)
        fp_smpl = f'{fp_pre}_{i:02d}{ext}'
        print(f'saving to {fp_smpl}')
        with open(fp_smpl, 'w') as f:
            f.write(
                json.dumps(nx.cytoscape_data(sample))
            )


if __name__ == '__main__':
    if len(sys.argv) not in [2, 3]:
        print('usage: python3 export_cyroscape_data.py <path> [<num_samples>]')

    if len(sys.argv) == 2:
        export(sys.argv[1])
    elif len(sys.argv) == 3:
        export_samples(sys.argv[1], int(sys.argv[2]))
