from contextgraph.util.graph import load_graph


def out_degree():
    G = load_graph()
    print(G.out_degree('pwc:method/lstm'))
    print(G.out_degree('uxv:context/1806.06827-SVM-703-906'))
