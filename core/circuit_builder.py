import pandas as pd
from core.network_adjustment import natural_sort_key, build_graph


def build_graph_from_cleaned_legs(cleaned_df: pd.DataFrame):
    usable_cleaned = cleaned_df[
        cleaned_df["Status"] != "All Observations Excluded"
    ].copy()

    usable_cleaned = usable_cleaned[
        usable_cleaned["Final_Normalized_Delta_Z"] != ""
    ].copy()

    edges = list(
        zip(
            usable_cleaned["From_Point"].astype(str).tolist(),
            usable_cleaned["To_Point"].astype(str).tolist(),
        )
    )

    graph = build_graph(edges)
    return graph, usable_cleaned


def get_all_available_points(graph: dict):
    return sorted(graph.keys(), key=natural_sort_key)


def get_next_candidate_points(graph: dict, current_path: list[str]):
    if not current_path:
        return []

    current_point = current_path[-1]
    used_points = set(current_path[:-1])

    candidates = [
        pt for pt in graph.get(current_point, set())
        if pt not in used_points
    ]

    return sorted(candidates, key=natural_sort_key)


def auto_extend_circuit(graph: dict, current_path: list[str], fixed_points: set[str]):
    """
    Auto-extend while there is exactly one valid continuation.
    Stop when:
    - dead end
    - fork
    - returning would require revisiting
    """
    if not current_path:
        return current_path, "No start point selected."

    path = current_path[:]

    while True:
        candidates = get_next_candidate_points(graph, path)

        if len(candidates) == 0:
            return path, "Reached dead end."

        if len(candidates) > 1:
            return path, "Reached fork. User selection required."

        next_point = candidates[0]
        path.append(next_point)

        # Stop automatically if we just reached a fixed point beyond the start
        if len(path) > 1 and next_point in fixed_points:
            return path, "Reached fixed point."

    return path, "Stopped."


def classify_circuit_path(path: list[str], fixed_points: set[str]):
    if not path or len(path) < 2:
        return "Invalid"

    start_fixed = path[0] in fixed_points
    end_fixed = path[-1] in fixed_points

    if start_fixed and end_fixed:
        return "Fixed to Fixed"
    if start_fixed or end_fixed:
        return "Fixed to Free"
    return "Unanchored"


def build_circuit_legs_df(path: list[str], cleaned_df: pd.DataFrame):
    if len(path) < 2:
        return pd.DataFrame()

    usable_cleaned = cleaned_df[
        cleaned_df["Status"] != "All Observations Excluded"
    ].copy()

    usable_cleaned = usable_cleaned[
        usable_cleaned["Final_Normalized_Delta_Z"] != ""
    ].copy()

    leg_rows = []

    for i in range(len(path) - 1):
        a = str(path[i])
        b = str(path[i + 1])

        match = usable_cleaned[
            (
                (usable_cleaned["From_Point"].astype(str) == a)
                & (usable_cleaned["To_Point"].astype(str) == b)
            )
            |
            (
                (usable_cleaned["From_Point"].astype(str) == b)
                & (usable_cleaned["To_Point"].astype(str) == a)
            )
        ]

        if match.empty:
            leg_rows.append(
                {
                    "From_Point": a,
                    "To_Point": b,
                    "Leg_ID": "|".join(sorted([a, b], key=natural_sort_key)),
                    "Observed_Delta_Z": "",
                    "Status": "Missing",
                }
            )
            continue

        row = match.iloc[0]
        from_point = str(row["From_Point"])
        to_point = str(row["To_Point"])
        observed = float(row["Final_Normalized_Delta_Z"])

        if from_point == a and to_point == b:
            dz = observed
        else:
            dz = -observed

        leg_rows.append(
            {
                "From_Point": a,
                "To_Point": b,
                "Leg_ID": "|".join(sorted([a, b], key=natural_sort_key)),
                "Observed_Delta_Z": round(dz, 4),
                "Status": "OK",
            }
        )

    return pd.DataFrame(leg_rows)