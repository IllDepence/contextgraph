from contextgraph.util.graph import load_graph


def out_degree():
    G = load_graph()
    print(G.out_degree('pwc:method/lstm'))
