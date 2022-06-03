import sys
import json
import networkx as nx
from contextgraph.util.graph import load_graph


def export(fp):
    print('loading graph')
    G = load_graph(shallow=True)
    print(f'saving to {fp}')
    with open(fp, 'w') as f:
        f.write(
            json.dumps(nx.cytoscape_data(G))
        )


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: python3 export_cyroscape_data.py <path>')
    export(sys.argv[1])
