from sage.graphs.graph import Graph

def save_adj_matrix(graph: Graph):
    """
    upload that file `output/adj_matrix.txt` in https://houseofgraphs.org/draw_graph as an
    adjacency matrix
    """
    adj_matrix = graph.adjacency_matrix()
    with open("output/adj_matrix.txt", "w") as f:
        for i in range(adj_matrix.nrows()):
            row = " ".join(str(adj_matrix[i, j]) for j in range(adj_matrix.ncols()))
            f.write(row + "\n")
