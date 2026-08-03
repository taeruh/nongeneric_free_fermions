def is_induced_ordered_path(graph, path) -> tuple[bool, bool]:
    """This is fine for Fendley's model when we order the indices according to their
    label, as induced paths in that graph must be ordered. It is not fine for other graphs
    like Fukai's model!"""
    length = len(path)
    is_induced = True
    is_path = True
    for i in range(length - 1):
        if not graph.has_edge(
            path[i],
            path[i + 1],
        ):
            is_path = False
            break
        for j in range(i + 2, length):
            if j == length:
                break
            if graph.has_edge(
                path[i],
                path[j],
            ):
                is_induced = False
                break
        if (not is_induced) or (not is_path):
            break
    return is_induced, is_path


def is_induced_path(graph, path) -> bool:
    """path must have at least 2 vertices"""
    subgraph = graph.subgraph(path)
    # subgraph.plot().save_image("output/subgraph.png")  # pyright: ignore
    # returns only the empty list if path has only a single vertex
    longest_path = subgraph.longest_path()  
    return longest_path.n_vertices() == len(path) and len(path) - 1 == subgraph.num_edges()
