def find_fixed_to_fixed_paths(graph, fixed_points, max_depth=50):
    paths = []

    def dfs(current, start, visited, path):
        if len(path) > max_depth:
            return

        for neighbor in graph.get(current, []):
            if neighbor in visited:
                continue

            new_path = path + [neighbor]

            # Found another fixed point
            if neighbor in fixed_points and neighbor != start:
                paths.append(new_path)
                continue

            dfs(neighbor, start, visited | {neighbor}, new_path)

    for fp in fixed_points:
        dfs(fp, fp, {fp}, [fp])

    # Remove duplicates (A→B same as B→A)
    unique = []
    seen = set()

    for p in paths:
        key = tuple(sorted(p))
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique