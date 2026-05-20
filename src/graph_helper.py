def is_induced_path(graph, path) -> tuple[bool, bool]:
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
