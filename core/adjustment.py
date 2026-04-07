import re
import numpy as np
import pandas as pd


CONTROL_REQUIRED_COLUMNS = ["PointID", "Elevation", "Fixed"]


def natural_sort_key(value):
    if pd.isna(value):
        return tuple()

    text = str(value)
    parts = re.split(r"(\d+)", text)

    key = []
    for part in parts:
        if part == "":
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))

    return tuple(key)


def clean_text_identifier(value):
    if pd.isna(value):
        return ""

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value).strip()


def validate_control_points(df: pd.DataFrame):
    errors = []
    warnings = []

    if df is None or df.empty:
        errors.append("Control_Points sheet is empty.")
        return pd.DataFrame(), errors, warnings

    missing = [col for col in CONTROL_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        errors.append(
            f"Missing required columns in Control_Points: {', '.join(missing)}"
        )
        return pd.DataFrame(), errors, warnings

    work = df[CONTROL_REQUIRED_COLUMNS].copy()

    blank_mask = work.isna().all(axis=1)
    blank_rows = work.index[blank_mask].tolist()
    if blank_rows:
        warnings.append(
            "Ignored completely blank rows in Control_Points at Excel rows: "
            + ", ".join(str(i + 2) for i in blank_rows)
        )
    work = work.loc[~blank_mask].copy()

    if work.empty:
        errors.append("Control_Points contains no usable rows.")
        return pd.DataFrame(), errors, warnings

    partial_blank_mask = work.isna().any(axis=1)
    partial_rows = work.index[partial_blank_mask].tolist()
    if partial_rows:
        warnings.append(
            "Discarded rows with blank required cells in Control_Points at Excel rows: "
            + ", ".join(str(i + 2) for i in partial_rows)
        )
    work = work.loc[~partial_blank_mask].copy()

    work["PointID"] = work["PointID"].apply(clean_text_identifier)
    work["Fixed"] = work["Fixed"].astype(str).str.strip().str.upper()

    empty_id_mask = work["PointID"] == ""
    if empty_id_mask.any():
        bad_rows = work.index[empty_id_mask].tolist()
        warnings.append(
            "Discarded rows with empty PointID in Control_Points at Excel rows: "
            + ", ".join(str(i + 2) for i in bad_rows)
        )
    work = work.loc[~empty_id_mask].copy()

    elevation_numeric = pd.to_numeric(work["Elevation"], errors="coerce")
    bad_elev_mask = elevation_numeric.isna()
    if bad_elev_mask.any():
        bad_rows = work.index[bad_elev_mask].tolist()
        warnings.append(
            "Discarded rows with non-numeric Elevation in Control_Points at Excel rows: "
            + ", ".join(str(i + 2) for i in bad_rows)
        )
    work = work.loc[~bad_elev_mask].copy()
    elevation_numeric = elevation_numeric.loc[~bad_elev_mask]

    valid_fixed_mask = work["Fixed"].isin(["Y", "N"])
    if (~valid_fixed_mask).any():
        bad_rows = work.index[~valid_fixed_mask].tolist()
        warnings.append(
            "Discarded rows with invalid Fixed value in Control_Points at Excel rows: "
            + ", ".join(str(i + 2) for i in bad_rows)
            + ". Use Y or N."
        )
    work = work.loc[valid_fixed_mask].copy()
    elevation_numeric = elevation_numeric.loc[valid_fixed_mask]

    if work.empty:
        errors.append("Control_Points contains no usable rows after validation.")
        return pd.DataFrame(), errors, warnings

    work["Elevation"] = elevation_numeric.astype(float)

    dupes = work.duplicated(subset=["PointID"], keep=False)
    if dupes.any():
        dup_rows = work.index[dupes].tolist()
        errors.append(
            "Duplicate PointID values found in Control_Points at Excel rows: "
            + ", ".join(str(i + 2) for i in dup_rows)
        )

    return work.reset_index(drop=True), errors, warnings


def _sort_df_by_point(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    work["_sort"] = work[column].apply(natural_sort_key)
    work = work.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    return work


def _build_graph(edges):
    graph = {}
    for from_point, to_point in edges:
        graph.setdefault(from_point, set()).add(to_point)
        graph.setdefault(to_point, set()).add(from_point)
    return graph


def _build_components(edges):
    graph = _build_graph(edges)
    visited = set()
    components = []

    for point in sorted(graph.keys(), key=natural_sort_key):
        if point in visited:
            continue

        stack = [point]
        component = set()

        while stack:
            current = stack.pop()
            if current in visited:
                continue

            visited.add(current)
            component.add(current)

            for neighbor in graph.get(current, set()):
                if neighbor not in visited:
                    stack.append(neighbor)

        components.append(component)

    return components


def _classify_section(start_point, end_point, start_fixed, end_fixed, component_adjustable):
    if start_point == end_point:
        return "Loop"

    if start_fixed and end_fixed:
        return "Fixed to Fixed"

    if start_fixed or end_fixed:
        return "Fixed to Free"

    if component_adjustable:
        return "Free to Free"

    return "Unanchored"


def _trace_sections_for_component(component_id, component_points, component_edges, fixed_points_in_component):
    """
    Break one connected component into chain-like sections between special nodes.

    Special nodes:
    - fixed points
    - degree != 2 nodes

    Returns rows for a Network Sections report.
    """
    graph = _build_graph(component_edges)
    component_adjustable = len(fixed_points_in_component) > 0

    degrees = {pt: len(graph.get(pt, set())) for pt in component_points}
    special_nodes = {
        pt for pt in component_points
        if pt in fixed_points_in_component or degrees.get(pt, 0) != 2
    }

    visited_edges = set()
    section_rows = []
    section_counter = 1

    def edge_key(a, b):
        return tuple(sorted((a, b), key=natural_sort_key))

    def leg_id(a, b):
        return "|".join(sorted((str(a), str(b)), key=natural_sort_key))

    # Trace chains starting from special nodes
    for start_node in sorted(special_nodes, key=natural_sort_key):
        neighbors = sorted(graph.get(start_node, set()), key=natural_sort_key)
        for neighbor in neighbors:
            ek = edge_key(start_node, neighbor)
            if ek in visited_edges:
                continue

            path_points = [start_node]
            path_legs = []
            prev = start_node
            current = neighbor

            visited_edges.add(ek)
            path_legs.append(leg_id(prev, current))

            while True:
                path_points.append(current)
                current_neighbors = sorted(graph.get(current, set()), key=natural_sort_key)

                if current in special_nodes and current != start_node:
                    end_node = current
                    break

                next_candidates = [n for n in current_neighbors if n != prev]

                if not next_candidates:
                    end_node = current
                    break

                next_node = next_candidates[0]
                ek2 = edge_key(current, next_node)
                if ek2 in visited_edges:
                    end_node = current
                    break

                visited_edges.add(ek2)
                path_legs.append(leg_id(current, next_node))
                prev, current = current, next_node

            start_fixed = start_node in fixed_points_in_component
            end_fixed = end_node in fixed_points_in_component

            section_rows.append(
                {
                    "Section_ID": f"C{component_id}-S{section_counter}",
                    "Component_ID": component_id,
                    "Start_Point": start_node,
                    "End_Point": end_node,
                    "Start_Type": "Fixed" if start_fixed else ("Junction/End" if degrees.get(start_node, 0) != 2 else "Intermediate"),
                    "End_Type": "Fixed" if end_fixed else ("Junction/End" if degrees.get(end_node, 0) != 2 else "Intermediate"),
                    "Section_Type": _classify_section(start_node, end_node, start_fixed, end_fixed, component_adjustable),
                    "Leg_Count": len(path_legs),
                    "Point_Count": len(path_points),
                    "Points_In_Order": " -> ".join(path_points),
                    "Legs_In_Order": ", ".join(path_legs),
                    "Adjustable": "Y" if component_adjustable else "N",
                }
            )
            section_counter += 1

    # Handle pure loops with no special nodes
    remaining_edges = []
    for a, neighbors in graph.items():
        for b in neighbors:
            ek = edge_key(a, b)
            if ek not in visited_edges:
                remaining_edges.append((a, b))

    if remaining_edges:
        used_loop_edges = set()
        for a, b in remaining_edges:
            ek = edge_key(a, b)
            if ek in used_loop_edges:
                continue

            path_points = [a]
            path_legs = []
            prev = a
            current = b
            used_loop_edges.add(ek)

            while True:
                path_points.append(current)
                path_legs.append(leg_id(prev, current))
                next_candidates = [n for n in sorted(graph.get(current, set()), key=natural_sort_key) if n != prev]
                if not next_candidates:
                    break

                next_node = next_candidates[0]
                ek2 = edge_key(current, next_node)
                if next_node == a:
                    path_points.append(a)
                    path_legs.append(leg_id(current, a))
                    used_loop_edges.add(ek2)
                    break
                if ek2 in used_loop_edges:
                    break

                used_loop_edges.add(ek2)
                prev, current = current, next_node

            section_rows.append(
                {
                    "Section_ID": f"C{component_id}-S{section_counter}",
                    "Component_ID": component_id,
                    "Start_Point": path_points[0],
                    "End_Point": path_points[-1],
                    "Start_Type": "Loop",
                    "End_Type": "Loop",
                    "Section_Type": "Loop" if component_adjustable else "Unanchored",
                    "Leg_Count": len(path_legs),
                    "Point_Count": len(path_points),
                    "Points_In_Order": " -> ".join(path_points),
                    "Legs_In_Order": ", ".join(path_legs),
                    "Adjustable": "Y" if component_adjustable else "N",
                }
            )
            section_counter += 1

    return section_rows


def _run_component_adjustment(component_points, component_cleaned_df, fixed_points_in_component):
    adjusted_elevations = {}
    point_sigmas = {}
    residual_rows = []

    component_points = sorted(component_points, key=natural_sort_key)
    unknown_points = [pt for pt in component_points if pt not in fixed_points_in_component]
    point_index = {pt: idx for idx, pt in enumerate(unknown_points)}

    if component_cleaned_df.empty:
        return adjusted_elevations, point_sigmas, residual_rows, "No usable cleaned legs in component."

    A_rows = []
    l_values = []
    leg_rows = []

    for _, row in component_cleaned_df.iterrows():
        from_point = str(row["From_Point"])
        to_point = str(row["To_Point"])
        observed_dz = float(row["Final_Normalized_Delta_Z"])

        a = np.zeros(len(unknown_points), dtype=float)
        rhs = observed_dz

        if from_point in point_index:
            a[point_index[from_point]] -= 1.0
        elif from_point in fixed_points_in_component:
            rhs += fixed_points_in_component[from_point]

        if to_point in point_index:
            a[point_index[to_point]] += 1.0
        elif to_point in fixed_points_in_component:
            rhs -= fixed_points_in_component[to_point]

        # Stochastic weighting is intentionally disabled for now.
        # We solve using an unweighted model (all observations treated equally),
        # and report each observation with a unit weight.
        weight = 1.0

        A_rows.append(a)
        l_values.append(rhs)

        leg_rows.append(
            {
                "From_Point": from_point,
                "To_Point": to_point,
                "Leg_ID": row["Leg_ID"],
                "Final_Normalized_Delta_Z": observed_dz,
                "Weight": weight,
            }
        )

    for point, elev in fixed_points_in_component.items():
        adjusted_elevations[point] = elev
        point_sigmas[point] = 0.0

    if len(unknown_points) == 0:
        for row in leg_rows:
            adjusted_dz = adjusted_elevations[row["To_Point"]] - adjusted_elevations[row["From_Point"]]
            residual = adjusted_dz - row["Final_Normalized_Delta_Z"]
            residual_rows.append(
                {
                    "From_Point": row["From_Point"],
                    "To_Point": row["To_Point"],
                    "Leg_ID": row["Leg_ID"],
                    "Final_Normalized_Delta_Z": row["Final_Normalized_Delta_Z"],
                    "Adjusted_Delta_Z": round(adjusted_dz, 4),
                    "Residual": round(residual, 4),
                    "Weight": round(row["Weight"], 6),
                    "Used_In_Adjustment": "Y",
                }
            )
        return adjusted_elevations, point_sigmas, residual_rows, ""

    A = np.array(A_rows, dtype=float)
    l = np.array(l_values, dtype=float).reshape(-1, 1)

    N = A.T @ A
    u = A.T @ l

    try:
        x_hat = np.linalg.solve(N, u)
    except np.linalg.LinAlgError:
        return adjusted_elevations, point_sigmas, residual_rows, (
            "Adjustment matrix is singular for this connected component."
        )

    x_hat = x_hat.flatten()

    for point, idx in point_index.items():
        adjusted_elevations[point] = float(x_hat[idx])

    v = (A @ x_hat.reshape(-1, 1)) - l
    v_flat = v.flatten()

    dof = len(l_values) - len(unknown_points)
    if dof > 0:
        vt_v = (v.T @ v).item()
        sigma0_sq = vt_v / dof
        covariance = sigma0_sq * np.linalg.inv(N)
        for point, idx in point_index.items():
            point_sigmas[point] = float(np.sqrt(covariance[idx, idx]))
    else:
        for point in unknown_points:
            point_sigmas[point] = ""

    for idx, row in enumerate(leg_rows):
        adjusted_dz = row["Final_Normalized_Delta_Z"] + float(v_flat[idx])
        residual_rows.append(
            {
                "From_Point": row["From_Point"],
                "To_Point": row["To_Point"],
                "Leg_ID": row["Leg_ID"],
                "Final_Normalized_Delta_Z": row["Final_Normalized_Delta_Z"],
                "Adjusted_Delta_Z": round(adjusted_dz, 4),
                "Residual": round(float(v_flat[idx]), 4),
                "Weight": round(row["Weight"], 6),
                "Used_In_Adjustment": "Y",
            }
        )

    return adjusted_elevations, point_sigmas, residual_rows, ""


def run_least_squares_adjustment(cleaned_df: pd.DataFrame, control_df: pd.DataFrame):
    errors = []
    warnings = []

    adjusted_points_df = pd.DataFrame()
    observation_residuals_df = pd.DataFrame()
    control_checks_df = pd.DataFrame()
    connectivity_report_df = pd.DataFrame()
    sections_report_df = pd.DataFrame()

    if cleaned_df.empty:
        errors.append("No cleaned leg means available for adjustment.")
        return (
            adjusted_points_df,
            observation_residuals_df,
            control_checks_df,
            connectivity_report_df,
            sections_report_df,
            errors,
            warnings,
        )

    if control_df.empty:
        errors.append("No valid control points available for adjustment.")
        return (
            adjusted_points_df,
            observation_residuals_df,
            control_checks_df,
            connectivity_report_df,
            sections_report_df,
            errors,
            warnings,
        )

    usable_cleaned = cleaned_df[
        cleaned_df["Status"] != "All Observations Excluded"
    ].copy()

    usable_cleaned = usable_cleaned[
        usable_cleaned["Final_Normalized_Delta_Z"] != ""
    ].copy()

    if usable_cleaned.empty:
        errors.append("No usable cleaned legs are available for adjustment.")
        return (
            adjusted_points_df,
            observation_residuals_df,
            control_checks_df,
            connectivity_report_df,
            sections_report_df,
            errors,
            warnings,
        )

    control_map = {
        row["PointID"]: {
            "Elevation": float(row["Elevation"]),
            "Fixed": row["Fixed"],
        }
        for _, row in control_df.iterrows()
    }

    fixed_points = {
        point_id: info["Elevation"]
        for point_id, info in control_map.items()
        if info["Fixed"] == "Y"
    }

    if not fixed_points:
        errors.append("At least one control point must have Fixed = Y.")
        return (
            adjusted_points_df,
            observation_residuals_df,
            control_checks_df,
            connectivity_report_df,
            sections_report_df,
            errors,
            warnings,
        )

    edge_pairs = list(
        zip(
            usable_cleaned["From_Point"].astype(str).tolist(),
            usable_cleaned["To_Point"].astype(str).tolist(),
        )
    )

    components = _build_components(edge_pairs)

    connectivity_rows = []
    all_adjusted_elevations = {}
    all_point_sigmas = {}
    all_residual_rows = []
    all_section_rows = []

    for idx, component in enumerate(components, start=1):
        component_points = sorted(component, key=natural_sort_key)

        fixed_in_component = {
            point: elev for point, elev in fixed_points.items() if point in component
        }

        component_cleaned = usable_cleaned[
            usable_cleaned["From_Point"].astype(str).isin(component)
            & usable_cleaned["To_Point"].astype(str).isin(component)
        ].copy()

        component_edges = list(
            zip(
                component_cleaned["From_Point"].astype(str).tolist(),
                component_cleaned["To_Point"].astype(str).tolist(),
            )
        )

        point_count = len(component_points)
        fixed_point_count = len(fixed_in_component)
        observation_count = len(component_cleaned)
        unknown_count = point_count - fixed_point_count
        degrees_of_freedom = observation_count - unknown_count

        if fixed_in_component:
            status = "Adjustable"
        else:
            status = "Not Adjustable"

        redundancy_status = "Redundant" if degrees_of_freedom > 0 else "Exact Solution"

        connectivity_rows.append(
            {
                "Component_ID": idx,
                "Point_Count": point_count,
                "Observation_Count": observation_count,
                "Unknown_Count": unknown_count,
                "Degrees_Of_Freedom": degrees_of_freedom,
                "Fixed_Point_Count": fixed_point_count,
                "Fixed_Points": ", ".join(sorted(fixed_in_component.keys(), key=natural_sort_key)) if fixed_in_component else "—",
                "Status": status,
                "Redundancy_Status": redundancy_status if fixed_in_component else "Not Adjustable",
                "Points": ", ".join(component_points),
            }
        )

        all_section_rows.extend(
            _trace_sections_for_component(
                component_id=idx,
                component_points=component_points,
                component_edges=component_edges,
                fixed_points_in_component=fixed_in_component,
            )
        )

        if not fixed_in_component:
            warnings.append(
                f"Connected component {idx} has no fixed control point and was not adjusted."
            )
            continue

        (
            adjusted_elevations,
            point_sigmas,
            residual_rows,
            component_error,
        ) = _run_component_adjustment(
            component_points,
            component_cleaned,
            fixed_in_component,
        )

        if component_error:
            warnings.append(
                f"Connected component {idx} could not be adjusted: {component_error}"
            )
            continue

        all_adjusted_elevations.update(adjusted_elevations)
        all_point_sigmas.update(point_sigmas)
        all_residual_rows.extend(residual_rows)

    connectivity_report_df = pd.DataFrame(connectivity_rows)
    sections_report_df = pd.DataFrame(all_section_rows)

    if not sections_report_df.empty:
        sections_report_df["_section_sort"] = sections_report_df["Section_ID"].apply(natural_sort_key)
        sections_report_df = (
            sections_report_df.sort_values(["Component_ID", "_section_sort"])
            .drop(columns=["_section_sort"])
            .reset_index(drop=True)
        )

    all_network_points = sorted(
        set(cleaned_df["From_Point"].astype(str))
        | set(cleaned_df["To_Point"].astype(str)),
        key=natural_sort_key,
    )

    adjusted_point_rows = []
    for point in all_network_points:
        adjusted_elevation = all_adjusted_elevations.get(point, "")
        sigma = all_point_sigmas.get(point, "")
        control_elevation = ""
        fixed_flag = ""
        difference_to_control = ""

        if point in control_map:
            control_elevation = float(control_map[point]["Elevation"])
            fixed_flag = control_map[point]["Fixed"]
            if adjusted_elevation != "":
                difference_to_control = round(adjusted_elevation - control_elevation, 4)

        adjusted_point_rows.append(
            {
                "Point_ID": point,
                "Adjusted_Elevation": round(adjusted_elevation, 4) if adjusted_elevation != "" else "",
                "Sigma": round(sigma, 4) if sigma != "" else "",
                "Fixed": fixed_flag,
                "Control_Elevation": round(control_elevation, 4) if control_elevation != "" else "",
                "Difference_To_Control": difference_to_control if difference_to_control == "" else round(difference_to_control, 4),
            }
        )

    adjusted_points_df = pd.DataFrame(adjusted_point_rows)
    adjusted_points_df = _sort_df_by_point(adjusted_points_df, "Point_ID")

    residual_lookup = {
        (row["Leg_ID"], row["From_Point"], row["To_Point"]): row for row in all_residual_rows
    }

    residual_rows = []
    for _, row in cleaned_df.iterrows():
        leg_id = row["Leg_ID"]
        from_point = str(row["From_Point"])
        to_point = str(row["To_Point"])

        key = (leg_id, from_point, to_point)

        if key in residual_lookup:
            residual_rows.append(residual_lookup[key])
        else:
            residual_rows.append(
                {
                    "From_Point": from_point,
                    "To_Point": to_point,
                    "Leg_ID": leg_id,
                    "Final_Normalized_Delta_Z": row["Final_Normalized_Delta_Z"],
                    "Adjusted_Delta_Z": "",
                    "Residual": "",
                    "Weight": "",
                    "Used_In_Adjustment": "N",
                }
            )

    observation_residuals_df = pd.DataFrame(residual_rows)
    if not observation_residuals_df.empty:
        observation_residuals_df["_leg_sort"] = observation_residuals_df["Leg_ID"].apply(natural_sort_key)
        observation_residuals_df = (
            observation_residuals_df.sort_values("_leg_sort")
            .drop(columns=["_leg_sort"])
            .reset_index(drop=True)
        )

    control_check_rows = []
    for _, row in control_df.iterrows():
        point = row["PointID"]
        control_elevation = float(row["Elevation"])
        fixed_flag = row["Fixed"]

        if point in all_adjusted_elevations:
            adjusted_elevation = all_adjusted_elevations[point]
            difference = adjusted_elevation - control_elevation
        else:
            adjusted_elevation = ""
            difference = ""

        control_check_rows.append(
            {
                "Point_ID": point,
                "Control_Elevation": round(control_elevation, 4),
                "Adjusted_Elevation": round(adjusted_elevation, 4) if adjusted_elevation != "" else "",
                "Fixed": fixed_flag,
                "Difference": round(difference, 4) if difference != "" else "",
            }
        )

    control_checks_df = pd.DataFrame(control_check_rows)
    control_checks_df = _sort_df_by_point(control_checks_df, "Point_ID")

    if not any(row["Status"] == "Adjustable" for row in connectivity_rows):
        errors.append(
            "No connected component contains a fixed control point, so no adjustment could be performed."
        )

    return (
        adjusted_points_df,
        observation_residuals_df,
        control_checks_df,
        connectivity_report_df,
        sections_report_df,
        errors,
        warnings,
    )
