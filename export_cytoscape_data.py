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
    G = load_graph()  # shallow=True
    print('generating samples')
    samples_pos, samples_neg = get_pair_graphs(int(num_samples/2), G)
    for posneg, samples in [
        ('pos', samples_pos),
        ('neg', samples_neg),
    ]:
        for i, sample in enumerate(samples):
            fp_pre, ext = os.path.splitext(fp)
            assert(ext == '.json')
            fp_smpl = f'{fp_pre}_{posneg}_{i:02d}_graph{ext}'
            fp_edge = f'{fp_pre}_{posneg}_{i:02d}_prediction_edge{ext}'
            print(f'saving graph to {fp_smpl}')
            with open(fp_smpl, 'w') as f:
                f.write(
                    json.dumps(nx.cytoscape_data(sample['graph']))
                )
            print(f'saving prediction edge to {fp_edge}')
            sample['prediction_edge']['cooc_pprs'] = list(
                sample['prediction_edge']['cooc_pprs']
            )
            with open(fp_edge, 'w') as f:
                json.dump(sample['prediction_edge'], f)


if __name__ == '__main__':
    if len(sys.argv) not in [2, 3]:
        print('usage: python3 export_cyroscape_data.py <path> [<num_samples>]')

    if len(sys.argv) == 2:
        export_cyto(sys.argv[1])
    elif len(sys.argv) == 3:
        export_samples_cyto(sys.argv[1], int(sys.argv[2]))
