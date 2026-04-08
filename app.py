from flask import Flask, render_template, request, send_file
import pandas as pd
from io import StringIO, BytesIO
import json

from core.leg_computation import validate_field_data, compute_legs
from core.repeated_leg_analysis import analyze_repeated_legs
from core.cleaned_leg_means import build_cleaned_leg_means
from core.control_points import (
    validate_control_points,
    update_control_fixed_flags,
    apply_anchor_elevation,
)
from core.network_adjustment import natural_sort_key, run_network_adjustment
from core.circuit_builder import (
    build_graph_from_cleaned_legs,
    get_all_available_points,
    get_next_candidate_points,
    auto_extend_circuit,
    classify_circuit_path,
)
from core.circuit_adjustment import compute_circuit_adjustment
from core.export_helpers import export_analysis_workbook

app = Flask(__name__)


def sort_summary_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    work = summary_df.copy()
    work["_leg_sort"] = work["Leg_ID"].apply(natural_sort_key)
    work = work.sort_values(by="_leg_sort").drop(columns=["_leg_sort"]).reset_index(drop=True)
    return work


def sort_decision_df(decision_df: pd.DataFrame) -> pd.DataFrame:
    if decision_df.empty:
        return decision_df

    work = decision_df.copy()
    work["_leg_sort"] = work["Leg_ID"].apply(natural_sort_key)
    work["_run_sort"] = work["Run_ID"].apply(natural_sort_key)
    work["_from_sort"] = work["From_Point"].apply(natural_sort_key)
    work["_to_sort"] = work["To_Point"].apply(natural_sort_key)

    work = (
        work.sort_values(
            by=["_leg_sort", "_run_sort", "_from_sort", "_to_sort", "Normalized_Delta_Z"]
        )
        .drop(columns=["_leg_sort", "_run_sort", "_from_sort", "_to_sort"])
        .reset_index(drop=True)
    )
    return work


def sort_cleaned_df(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    if cleaned_df.empty:
        return cleaned_df

    work = cleaned_df.copy()
    work["_leg_sort"] = work["Leg_ID"].apply(natural_sort_key)
    work = work.sort_values(by="_leg_sort").drop(columns=["_leg_sort"]).reset_index(drop=True)
    return work


def make_decision_row_key(row) -> str:
    return (
        f"{row['Leg_ID']}||{row['Run_ID']}||{row['From_Point']}||"
        f"{row['To_Point']}"
    )


def recompute_after_exclusions(
    original_decision_df: pd.DataFrame, tolerance: float, selected_keys: set[str]
):
    work = original_decision_df.copy()

    work["Row_Key"] = work.apply(make_decision_row_key, axis=1)
    work["User_Excluded"] = work["Row_Key"].isin(selected_keys)

    summary_rows = []
    updated_rows = []

    for leg_id, group in work.groupby("Leg_ID", sort=False):
        active_group = group.loc[~group["User_Excluded"]].copy()

        if active_group.empty:
            reference_value = None
        else:
            diffs = active_group["Normalized_Delta_Z"].astype(float).tolist()
            observation_count = len(diffs)
            reference_value = float(pd.Series(diffs).median())
            residuals = [abs(d - reference_value) for d in diffs]
            max_residual = max(residuals) if residuals else 0.0

            if observation_count == 1:
                status = "Single Observation"
            elif observation_count == 2:
                if abs(diffs[0] - diffs[1]) <= tolerance:
                    status = "Within Tolerance"
                else:
                    status = "Review Required"
            else:
                if max_residual <= tolerance:
                    status = "Within Tolerance"
                else:
                    status = "Suspect Observation"

            summary_rows.append(
                {
                    "Leg_ID": leg_id,
                    "Observation_Count": observation_count,
                    "Reference_Value": round(reference_value, 4),
                    "Maximum_Residual": round(max_residual, 4),
                    "Status": status,
                }
            )

        for _, row in group.iterrows():
            row_dict = row.to_dict()
            diff = float(row_dict["Normalized_Delta_Z"])

            if reference_value is None:
                row_dict["Reference_Value"] = ""
                row_dict["Residual_From_Reference"] = ""
                row_dict["Decision"] = "Excluded by User"
            else:
                residual = abs(diff - reference_value)
                row_dict["Reference_Value"] = round(reference_value, 4)
                row_dict["Residual_From_Reference"] = round(residual, 4)

                if row_dict["User_Excluded"]:
                    row_dict["Decision"] = "Excluded by User"
                else:
                    active_diffs = active_group["Normalized_Delta_Z"].astype(float).tolist()
                    active_count = len(active_diffs)

                    if active_count == 1:
                        row_dict["Decision"] = "Single Observation"
                    elif active_count == 2 and abs(active_diffs[0] - active_diffs[1]) > tolerance:
                        row_dict["Decision"] = "Pending Review"
                    else:
                        row_dict["Decision"] = "Accepted" if residual <= tolerance else "Suspect"

            updated_rows.append(row_dict)

    summary_df = pd.DataFrame(summary_rows)
    updated_decision_df = pd.DataFrame(updated_rows)

    summary_df = sort_summary_df(summary_df)
    updated_decision_df = sort_decision_df(updated_decision_df)
    updated_decision_df = updated_decision_df.drop(columns=["Row_Key"], errors="ignore")

    return summary_df, updated_decision_df


def parse_json_df(json_text: str) -> pd.DataFrame:
    if not json_text:
        return pd.DataFrame()
    return pd.read_json(StringIO(json_text))


def parse_saved_circuits(json_text: str):
    if not json_text:
        return []
    return normalize_saved_circuits(json.loads(json_text))


def renumber_saved_circuits(saved_circuits: list[dict]) -> list[dict]:
    for idx, circuit in enumerate(saved_circuits, start=1):
        circuit["Circuit_ID"] = f"CIR-{idx}"
    return saved_circuits


def normalize_saved_circuits(saved_circuits: list[dict]) -> list[dict]:
    normalized = []
    for idx, circuit in enumerate(saved_circuits, start=1):
        path = [str(point) for point in circuit.get("Path", [])]
        normalized.append(
            {
                "Circuit_ID": str(circuit.get("Circuit_ID") or f"CIR-{idx}"),
                "Path": path,
            }
        )
    return normalized


def build_unassigned_points(
    all_points: list[str], current_circuit_path: list[str], saved_circuits: list[dict]
):
    current_points = {str(point) for point in current_circuit_path}
    saved_points = {
        str(point)
        for circuit in saved_circuits
        for point in circuit.get("Path", [])
    }

    return {
        "not_in_current": [point for point in all_points if point not in current_points],
        "not_in_saved": [point for point in all_points if point not in saved_points],
        "not_in_any": [
            point for point in all_points if point not in current_points and point not in saved_points
        ],
    }


def build_template_workbook(sheet_name: str, columns: list[str], filename: str):
    output = BytesIO()
    df = pd.DataFrame(columns=columns)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def run_network_pipeline(cleaned_df, control_df):
    return run_network_adjustment(cleaned_df, control_df)


def run_circuit_pipeline(saved_circuits, cleaned_df, control_df):
    return compute_circuit_adjustment(saved_circuits, cleaned_df, control_df)


def build_exclusion_state_file(selected_keys: set[str]):
    output = BytesIO()
    excluded_observations = []
    for key in sorted(selected_keys):
        parts = str(key).split("||")
        if len(parts) < 4:
            continue
        excluded_observations.append(
            {
                "leg_id": parts[0],
                "run_id": parts[1],
                "from_point": parts[2],
                "to_point": parts[3],
            }
        )

    payload = {
        "excluded_observations": excluded_observations,
    }
    output.write(json.dumps(payload, indent=2).encode("utf-8"))
    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name="exclusion_state.json",
        mimetype="application/json",
    )


def build_circuits_state_file(saved_circuits: list[dict]):
    output = BytesIO()
    payload = {
        "saved_circuits": normalize_saved_circuits(saved_circuits),
    }
    output.write(json.dumps(payload, indent=2).encode("utf-8"))
    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name="saved_circuits.json",
        mimetype="application/json",
    )


def build_unanchored_circuit_options(saved_circuits: list[dict], control_df: pd.DataFrame):
    if control_df is None or control_df.empty or not saved_circuits:
        return []

    fixed_points = set(
        control_df.loc[
            control_df["Fixed"].astype(str).str.upper() == "Y",
            "PointID",
        ].astype(str).tolist()
    )

    options = []
    for circuit in saved_circuits:
        path = [str(point) for point in circuit.get("Path", [])]
        if len(path) < 2:
            continue

        circuit_type = classify_circuit_path(path, fixed_points)
        if circuit_type != "Unanchored":
            continue

        unique_points = list(dict.fromkeys(path))
        options.append(
            {
                "circuit_id": str(circuit.get("Circuit_ID", "")),
                "points": unique_points,
            }
        )

    return options


def parse_exclusion_state_file(uploaded_state_file):
    content = uploaded_state_file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("State file is not a JSON object.")

    normalized_keys = set()

    excluded_observations = payload.get("excluded_observations")
    if excluded_observations is not None:
        if not isinstance(excluded_observations, list):
            raise ValueError("State file field 'excluded_observations' must be a list.")

        for row in excluded_observations:
            if not isinstance(row, dict):
                continue
            leg_id = str(row.get("leg_id", ""))
            run_id = str(row.get("run_id", ""))
            from_point = str(row.get("from_point", ""))
            to_point = str(row.get("to_point", ""))
            if leg_id and run_id and from_point and to_point:
                normalized_keys.add(f"{leg_id}||{run_id}||{from_point}||{to_point}")
    else:
        # Backward compatibility with older exclusion state files.
        excluded_rows = payload.get("excluded_rows", [])
        if not isinstance(excluded_rows, list):
            raise ValueError("State file field 'excluded_rows' must be a list.")

        for value in excluded_rows:
            key = str(value)
            parts = key.split("||")
            if len(parts) >= 4:
                normalized_keys.add("||".join(parts[:4]))
            else:
                normalized_keys.add(key)

    return normalized_keys


def parse_saved_circuits_file(uploaded_state_file):
    content = uploaded_state_file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("Circuits file is not a JSON object.")

    saved_circuits = payload.get("saved_circuits", [])
    if not isinstance(saved_circuits, list):
        raise ValueError("Circuits file field 'saved_circuits' must be a list.")

    return normalize_saved_circuits(saved_circuits)


@app.route("/", methods=["GET", "POST"])
def index():
    raw_data = None
    control_data = None
    leg_data = None
    summary_data = None
    decision_data = None
    cleaned_data = None

    adjusted_points_data = None
    observation_residuals_data = None
    control_checks_data = None
    connectivity_data = None
    sections_data = None

    circuit_summary_data = None
    circuit_legs_data = None
    circuit_elevations_data = None

    current_circuit_path = []
    current_circuit_candidates = []
    available_start_points = []
    current_circuit_type = ""
    current_circuit_message = ""
    saved_circuits = []
    unassigned_points = {"not_in_current": [], "not_in_saved": [], "not_in_any": []}

    errors = []
    warnings = []
    adjustment_messages = []
    success_message = None

    tolerance = 0.005
    residual_threshold = 0.005

    active_stage = "review"
    active_tab = "raw"
    active_adjustment_tab = "connectivity"
    adjustment_mode = "network"

    if request.method == "POST":
        action = request.form.get("action", "process")

        if action == "download_field_template":
            return build_template_workbook(
                sheet_name="Field_Elevations",
                columns=["RunID", "Sequence", "PointID", "Raw_Elevation"],
                filename="field_observations_template.xlsx",
            )

        if action == "download_control_template":
            return build_template_workbook(
                sheet_name="Control_Points",
                columns=["PointID", "Elevation", "Fixed"],
                filename="control_points_template.xlsx",
            )

        if action == "download_exclusion_state":
            selected_keys = set(request.form.getlist("exclude_row"))
            return build_exclusion_state_file(selected_keys)

        active_stage = request.form.get("active_stage", "review")
        active_tab = request.form.get("active_tab", "raw")
        active_adjustment_tab = request.form.get("active_adjustment_tab", "connectivity")
        adjustment_mode = request.form.get("adjustment_mode", "network")

        tolerance_raw = request.form.get("tolerance", "0.005")
        try:
            tolerance = float(tolerance_raw)
            if tolerance <= 0:
                errors.append("Tolerance must be greater than zero.")
        except ValueError:
            errors.append("Tolerance must be a numeric value.")

        residual_threshold_raw = request.form.get("residual_threshold", "0.005")
        try:
            residual_threshold = float(residual_threshold_raw)
            if residual_threshold < 0:
                errors.append("Residual threshold must be zero or greater.")
        except ValueError:
            errors.append("Residual threshold must be a numeric value.")

        uploaded_file = request.files.get("excel_file")
        uploaded_state_file = request.files.get("exclusion_state_file")
        uploaded_circuits_file = request.files.get("circuits_state_file")

        raw_json = request.form.get("raw_json", "")
        control_json = request.form.get("control_json", "")
        leg_json = request.form.get("leg_json", "")
        summary_json = request.form.get("summary_json", "")
        decision_json = request.form.get("decision_json", "")
        cleaned_json = request.form.get("cleaned_json", "")
        saved_circuits_json = request.form.get("saved_circuits_json", "[]")
        current_circuit_path_json = request.form.get("current_circuit_path_json", "[]")

        saved_circuits = parse_saved_circuits(saved_circuits_json)
        current_circuit_path = json.loads(current_circuit_path_json) if current_circuit_path_json else []

        selected_exclusions = set(request.form.getlist("exclude_row"))

        try:
            if action == "export_excel":
                if not raw_json:
                    errors.append("No analysis data found to export.")
                elif not errors:
                    raw_df = parse_json_df(raw_json)
                    control_df = parse_json_df(control_json)
                    leg_df = parse_json_df(leg_json)
                    summary_df = parse_json_df(summary_json)
                    decision_df = parse_json_df(decision_json)
                    cleaned_df = parse_json_df(cleaned_json)

                    (
                        adjusted_points_df,
                        observation_residuals_df,
                        control_checks_df,
                        connectivity_df,
                        sections_df,
                        _adj_errors,
                        _adj_warnings,
                    ) = (
                        run_network_pipeline(cleaned_df, control_df)
                        if (
                            adjustment_mode == "network"
                            and not control_df.empty
                            and not cleaned_df.empty
                        )
                        else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], [])
                    )

                    (
                        circuit_summary_df,
                        circuit_legs_df,
                        circuit_elevations_df,
                        _circuit_errors,
                        _circuit_warnings,
                    ) = (
                        run_circuit_pipeline(saved_circuits, cleaned_df, control_df)
                        if adjustment_mode == "circuit" and not control_df.empty
                        else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], [])
                    )

                    workbook = export_analysis_workbook(
                        raw_df,
                        control_df,
                        leg_df,
                        summary_df,
                        decision_df,
                        cleaned_df,
                        adjusted_points_df,
                        observation_residuals_df,
                        control_checks_df,
                        connectivity_df,
                        sections_df,
                        circuit_summary_df,
                        circuit_legs_df,
                        circuit_elevations_df,
                    )

                    return send_file(
                        workbook,
                        as_attachment=True,
                        download_name="leveling_analysis.xlsx",
                        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

            elif action == "apply_fixed_points":
                raw_df = parse_json_df(raw_json)
                control_df = parse_json_df(control_json)
                leg_df = parse_json_df(leg_json)
                summary_df = parse_json_df(summary_json)
                decision_df = parse_json_df(decision_json)
                cleaned_df = parse_json_df(cleaned_json)

                point_ids = request.form.getlist("control_point_id")
                fixed_values = request.form.getlist("control_fixed")

                control_df, fixed_errors = update_control_fixed_flags(control_df, point_ids, fixed_values)
                errors.extend(fixed_errors)

                raw_data = raw_df.to_dict(orient="records")
                control_data = control_df.to_dict(orient="records")
                leg_data = leg_df.to_dict(orient="records")
                summary_data = summary_df.to_dict(orient="records")
                decision_data = decision_df.to_dict(orient="records")
                cleaned_data = cleaned_df.to_dict(orient="records")

                if not errors and not control_df.empty and not cleaned_df.empty:
                    if adjustment_mode == "network":
                        (
                            adjusted_points_df,
                            observation_residuals_df,
                            control_checks_df,
                            connectivity_df,
                            sections_df,
                            adj_errors,
                            adj_warnings,
                        ) = run_network_pipeline(cleaned_df, control_df)

                        adjustment_messages.extend(adj_errors)
                        adjustment_messages.extend(adj_warnings)

                        adjusted_points_data = adjusted_points_df.to_dict(orient="records")
                        observation_residuals_data = observation_residuals_df.to_dict(orient="records")
                        control_checks_data = control_checks_df.to_dict(orient="records")
                        connectivity_data = connectivity_df.to_dict(orient="records")
                        sections_data = sections_df.to_dict(orient="records")
                    else:
                        (
                            circuit_summary_df,
                            circuit_legs_df,
                            circuit_elevations_df,
                            circuit_errors,
                            circuit_warnings,
                        ) = run_circuit_pipeline(saved_circuits, cleaned_df, control_df)

                        adjustment_messages.extend(circuit_errors)
                        adjustment_messages.extend(circuit_warnings)

                        circuit_summary_data = circuit_summary_df.to_dict(orient="records")
                        circuit_legs_data = circuit_legs_df.to_dict(orient="records")
                        circuit_elevations_data = circuit_elevations_df.to_dict(orient="records")

                        graph, _usable_cleaned = build_graph_from_cleaned_legs(cleaned_df)
                        available_start_points = get_all_available_points(graph)
                        fixed_points = set(
                            control_df.loc[
                                control_df["Fixed"].astype(str).str.upper() == "Y",
                                "PointID"
                            ].astype(str).tolist()
                        )
                        current_circuit_candidates = get_next_candidate_points(graph, current_circuit_path)
                        current_circuit_type = classify_circuit_path(current_circuit_path, fixed_points) if len(current_circuit_path) >= 2 else ""

                    success_message = "Fixed points updated successfully."

                active_stage = "adjustment"

            elif action == "download_circuits_state":
                return build_circuits_state_file(saved_circuits)

            elif action == "apply_uploaded_circuits":
                raw_df = parse_json_df(raw_json)
                control_df = parse_json_df(control_json)
                leg_df = parse_json_df(leg_json)
                summary_df = parse_json_df(summary_json)
                decision_df = parse_json_df(decision_json)
                cleaned_df = parse_json_df(cleaned_json)

                raw_data = raw_df.to_dict(orient="records")
                control_data = control_df.to_dict(orient="records")
                leg_data = leg_df.to_dict(orient="records")
                summary_data = summary_df.to_dict(orient="records")
                decision_data = decision_df.to_dict(orient="records")
                cleaned_data = cleaned_df.to_dict(orient="records")

                graph, _usable_cleaned = build_graph_from_cleaned_legs(cleaned_df)
                available_start_points = get_all_available_points(graph)

                if not uploaded_circuits_file or uploaded_circuits_file.filename == "":
                    current_circuit_message = "Please select a circuits JSON file."
                else:
                    saved_circuits = parse_saved_circuits_file(uploaded_circuits_file)
                    current_circuit_path = []
                    current_circuit_candidates = []
                    current_circuit_type = ""
                    current_circuit_message = "Circuits imported successfully."

                (
                    circuit_summary_df,
                    circuit_legs_df,
                    circuit_elevations_df,
                    circuit_errors,
                    circuit_warnings,
                ) = run_circuit_pipeline(saved_circuits, cleaned_df, control_df)

                adjustment_messages.extend(circuit_errors)
                adjustment_messages.extend(circuit_warnings)

                circuit_summary_data = circuit_summary_df.to_dict(orient="records")
                circuit_legs_data = circuit_legs_df.to_dict(orient="records")
                circuit_elevations_data = circuit_elevations_df.to_dict(orient="records")

                active_stage = "adjustment"
                active_adjustment_tab = "builder"

            elif action == "start_circuit":
                raw_df = parse_json_df(raw_json)
                control_df = parse_json_df(control_json)
                leg_df = parse_json_df(leg_json)
                summary_df = parse_json_df(summary_json)
                decision_df = parse_json_df(decision_json)
                cleaned_df = parse_json_df(cleaned_json)

                start_point = request.form.get("start_point", "").strip()

                raw_data = raw_df.to_dict(orient="records")
                control_data = control_df.to_dict(orient="records")
                leg_data = leg_df.to_dict(orient="records")
                summary_data = summary_df.to_dict(orient="records")
                decision_data = decision_df.to_dict(orient="records")
                cleaned_data = cleaned_df.to_dict(orient="records")

                graph, _usable_cleaned = build_graph_from_cleaned_legs(cleaned_df)
                available_start_points = get_all_available_points(graph)

                fixed_points = set(
                    control_df.loc[
                        control_df["Fixed"].astype(str).str.upper() == "Y",
                        "PointID"
                    ].astype(str).tolist()
                )

                if start_point:
                    current_circuit_path = [start_point]
                    current_circuit_path, current_circuit_message = auto_extend_circuit(
                        graph, current_circuit_path, fixed_points
                    )
                    current_circuit_candidates = get_next_candidate_points(graph, current_circuit_path)
                    current_circuit_type = classify_circuit_path(current_circuit_path, fixed_points) if len(current_circuit_path) >= 2 else ""
                else:
                    current_circuit_message = "Select a start point."

                (
                    circuit_summary_df,
                    circuit_legs_df,
                    circuit_elevations_df,
                    circuit_errors,
                    circuit_warnings,
                ) = run_circuit_pipeline(saved_circuits, cleaned_df, control_df)

                adjustment_messages.extend(circuit_errors)
                adjustment_messages.extend(circuit_warnings)

                circuit_summary_data = circuit_summary_df.to_dict(orient="records")
                circuit_legs_data = circuit_legs_df.to_dict(orient="records")
                circuit_elevations_data = circuit_elevations_df.to_dict(orient="records")

                active_stage = "adjustment"
                active_adjustment_tab = "builder"

            elif action == "choose_next_point":
                raw_df = parse_json_df(raw_json)
                control_df = parse_json_df(control_json)
                leg_df = parse_json_df(leg_json)
                summary_df = parse_json_df(summary_json)
                decision_df = parse_json_df(decision_json)
                cleaned_df = parse_json_df(cleaned_json)

                next_point = request.form.get("next_point", "").strip()

                raw_data = raw_df.to_dict(orient="records")
                control_data = control_df.to_dict(orient="records")
                leg_data = leg_df.to_dict(orient="records")
                summary_data = summary_df.to_dict(orient="records")
                decision_data = decision_df.to_dict(orient="records")
                cleaned_data = cleaned_df.to_dict(orient="records")

                graph, _usable_cleaned = build_graph_from_cleaned_legs(cleaned_df)
                available_start_points = get_all_available_points(graph)

                fixed_points = set(
                    control_df.loc[
                        control_df["Fixed"].astype(str).str.upper() == "Y",
                        "PointID"
                    ].astype(str).tolist()
                )

                if next_point:
                    current_circuit_path.append(next_point)
                    current_circuit_path, current_circuit_message = auto_extend_circuit(
                        graph, current_circuit_path, fixed_points
                    )

                current_circuit_candidates = get_next_candidate_points(graph, current_circuit_path)
                current_circuit_type = classify_circuit_path(current_circuit_path, fixed_points) if len(current_circuit_path) >= 2 else ""

                (
                    circuit_summary_df,
                    circuit_legs_df,
                    circuit_elevations_df,
                    circuit_errors,
                    circuit_warnings,
                ) = run_circuit_pipeline(saved_circuits, cleaned_df, control_df)

                adjustment_messages.extend(circuit_errors)
                adjustment_messages.extend(circuit_warnings)

                circuit_summary_data = circuit_summary_df.to_dict(orient="records")
                circuit_legs_data = circuit_legs_df.to_dict(orient="records")
                circuit_elevations_data = circuit_elevations_df.to_dict(orient="records")

                active_stage = "adjustment"
                active_adjustment_tab = "builder"

            elif action == "undo_circuit_point":
                raw_df = parse_json_df(raw_json)
                control_df = parse_json_df(control_json)
                leg_df = parse_json_df(leg_json)
                summary_df = parse_json_df(summary_json)
                decision_df = parse_json_df(decision_json)
                cleaned_df = parse_json_df(cleaned_json)

                raw_data = raw_df.to_dict(orient="records")
                control_data = control_df.to_dict(orient="records")
                leg_data = leg_df.to_dict(orient="records")
                summary_data = summary_df.to_dict(orient="records")
                decision_data = decision_df.to_dict(orient="records")
                cleaned_data = cleaned_df.to_dict(orient="records")

                if current_circuit_path:
                    current_circuit_path = current_circuit_path[:-1]

                graph, _usable_cleaned = build_graph_from_cleaned_legs(cleaned_df)
                available_start_points = get_all_available_points(graph)

                fixed_points = set(
                    control_df.loc[
                        control_df["Fixed"].astype(str).str.upper() == "Y",
                        "PointID"
                    ].astype(str).tolist()
                )

                current_circuit_candidates = get_next_candidate_points(graph, current_circuit_path)
                current_circuit_type = classify_circuit_path(current_circuit_path, fixed_points) if len(current_circuit_path) >= 2 else ""
                current_circuit_message = "Last point removed."

                (
                    circuit_summary_df,
                    circuit_legs_df,
                    circuit_elevations_df,
                    circuit_errors,
                    circuit_warnings,
                ) = run_circuit_pipeline(saved_circuits, cleaned_df, control_df)

                adjustment_messages.extend(circuit_errors)
                adjustment_messages.extend(circuit_warnings)

                circuit_summary_data = circuit_summary_df.to_dict(orient="records")
                circuit_legs_data = circuit_legs_df.to_dict(orient="records")
                circuit_elevations_data = circuit_elevations_df.to_dict(orient="records")

                active_stage = "adjustment"
                active_adjustment_tab = "builder"

            elif action == "clear_circuit":
                raw_df = parse_json_df(raw_json)
                control_df = parse_json_df(control_json)
                leg_df = parse_json_df(leg_json)
                summary_df = parse_json_df(summary_json)
                decision_df = parse_json_df(decision_json)
                cleaned_df = parse_json_df(cleaned_json)

                raw_data = raw_df.to_dict(orient="records")
                control_data = control_df.to_dict(orient="records")
                leg_data = leg_df.to_dict(orient="records")
                summary_data = summary_df.to_dict(orient="records")
                decision_data = decision_df.to_dict(orient="records")
                cleaned_data = cleaned_df.to_dict(orient="records")

                current_circuit_path = []
                current_circuit_candidates = []
                current_circuit_type = ""
                current_circuit_message = "Circuit cleared."

                graph, _usable_cleaned = build_graph_from_cleaned_legs(cleaned_df)
                available_start_points = get_all_available_points(graph)

                (
                    circuit_summary_df,
                    circuit_legs_df,
                    circuit_elevations_df,
                    circuit_errors,
                    circuit_warnings,
                ) = run_circuit_pipeline(saved_circuits, cleaned_df, control_df)

                adjustment_messages.extend(circuit_errors)
                adjustment_messages.extend(circuit_warnings)

                circuit_summary_data = circuit_summary_df.to_dict(orient="records")
                circuit_legs_data = circuit_legs_df.to_dict(orient="records")
                circuit_elevations_data = circuit_elevations_df.to_dict(orient="records")

                active_stage = "adjustment"
                active_adjustment_tab = "builder"

            elif action == "save_circuit":
                raw_df = parse_json_df(raw_json)
                control_df = parse_json_df(control_json)
                leg_df = parse_json_df(leg_json)
                summary_df = parse_json_df(summary_json)
                decision_df = parse_json_df(decision_json)
                cleaned_df = parse_json_df(cleaned_json)

                raw_data = raw_df.to_dict(orient="records")
                control_data = control_df.to_dict(orient="records")
                leg_data = leg_df.to_dict(orient="records")
                summary_data = summary_df.to_dict(orient="records")
                decision_data = decision_df.to_dict(orient="records")
                cleaned_data = cleaned_df.to_dict(orient="records")

                graph, _usable_cleaned = build_graph_from_cleaned_legs(cleaned_df)
                available_start_points = get_all_available_points(graph)

                if len(current_circuit_path) < 2:
                    current_circuit_message = "Circuit must contain at least two points."
                else:
                    saved_circuits.append(
                        {
                            "Circuit_ID": f"CIR-{len(saved_circuits) + 1}",
                            "Path": current_circuit_path[:],
                        }
                    )
                    current_circuit_message = "Circuit saved."
                    current_circuit_path = []
                    current_circuit_candidates = []
                    current_circuit_type = ""

                (
                    circuit_summary_df,
                    circuit_legs_df,
                    circuit_elevations_df,
                    circuit_errors,
                    circuit_warnings,
                ) = run_circuit_pipeline(saved_circuits, cleaned_df, control_df)

                adjustment_messages.extend(circuit_errors)
                adjustment_messages.extend(circuit_warnings)

                circuit_summary_data = circuit_summary_df.to_dict(orient="records")
                circuit_legs_data = circuit_legs_df.to_dict(orient="records")
                circuit_elevations_data = circuit_elevations_df.to_dict(orient="records")

                active_stage = "adjustment"
                active_adjustment_tab = "builder"

            elif action == "delete_selected_circuits":
                raw_df = parse_json_df(raw_json)
                control_df = parse_json_df(control_json)
                leg_df = parse_json_df(leg_json)
                summary_df = parse_json_df(summary_json)
                decision_df = parse_json_df(decision_json)
                cleaned_df = parse_json_df(cleaned_json)

                raw_data = raw_df.to_dict(orient="records")
                control_data = control_df.to_dict(orient="records")
                leg_data = leg_df.to_dict(orient="records")
                summary_data = summary_df.to_dict(orient="records")
                decision_data = decision_df.to_dict(orient="records")
                cleaned_data = cleaned_df.to_dict(orient="records")

                selected_circuit_ids = set(request.form.getlist("delete_circuit_id"))
                if selected_circuit_ids:
                    saved_circuits = [
                        circuit for circuit in saved_circuits
                        if circuit.get("Circuit_ID") not in selected_circuit_ids
                    ]
                    saved_circuits = renumber_saved_circuits(saved_circuits)
                    current_circuit_message = "Selected circuits deleted."
                else:
                    current_circuit_message = "No circuits were selected for deletion."

                graph, _usable_cleaned = build_graph_from_cleaned_legs(cleaned_df)
                available_start_points = get_all_available_points(graph)
                current_circuit_candidates = get_next_candidate_points(graph, current_circuit_path)

                (
                    circuit_summary_df,
                    circuit_legs_df,
                    circuit_elevations_df,
                    circuit_errors,
                    circuit_warnings,
                ) = run_circuit_pipeline(saved_circuits, cleaned_df, control_df)

                adjustment_messages.extend(circuit_errors)
                adjustment_messages.extend(circuit_warnings)

                circuit_summary_data = circuit_summary_df.to_dict(orient="records")
                circuit_legs_data = circuit_legs_df.to_dict(orient="records")
                circuit_elevations_data = circuit_elevations_df.to_dict(orient="records")

                active_stage = "adjustment"
                active_adjustment_tab = "circuit_summary"

            elif action == "apply_circuit_anchor":
                raw_df = parse_json_df(raw_json)
                control_df = parse_json_df(control_json)
                leg_df = parse_json_df(leg_json)
                summary_df = parse_json_df(summary_json)
                decision_df = parse_json_df(decision_json)
                cleaned_df = parse_json_df(cleaned_json)

                anchor_circuit_id = request.form.get("anchor_circuit_id", "").strip()
                anchor_point_id = request.form.get("anchor_point_id", "").strip()
                anchor_elevation = request.form.get("anchor_elevation", "").strip()

                raw_data = raw_df.to_dict(orient="records")
                leg_data = leg_df.to_dict(orient="records")
                summary_data = summary_df.to_dict(orient="records")
                decision_data = decision_df.to_dict(orient="records")
                cleaned_data = cleaned_df.to_dict(orient="records")

                selected_circuit = next(
                    (
                        circuit for circuit in saved_circuits
                        if str(circuit.get("Circuit_ID", "")).strip() == anchor_circuit_id
                    ),
                    None,
                )

                if selected_circuit is None:
                    errors.append("Selected circuit was not found.")
                else:
                    selected_path = [str(point) for point in selected_circuit.get("Path", [])]
                    if anchor_point_id not in selected_path:
                        errors.append("Selected point is not in the selected circuit.")

                if not errors:
                    control_df, anchor_errors = apply_anchor_elevation(
                        control_df,
                        anchor_point_id,
                        anchor_elevation,
                    )
                    errors.extend(anchor_errors)

                control_data = control_df.to_dict(orient="records")

                (
                    circuit_summary_df,
                    circuit_legs_df,
                    circuit_elevations_df,
                    circuit_errors,
                    circuit_warnings,
                ) = run_circuit_pipeline(saved_circuits, cleaned_df, control_df)

                adjustment_messages.extend(circuit_errors)
                adjustment_messages.extend(circuit_warnings)

                circuit_summary_data = circuit_summary_df.to_dict(orient="records")
                circuit_legs_data = circuit_legs_df.to_dict(orient="records")
                circuit_elevations_data = circuit_elevations_df.to_dict(orient="records")

                graph, _usable_cleaned = build_graph_from_cleaned_legs(cleaned_df)
                available_start_points = get_all_available_points(graph)
                fixed_points = set(
                    control_df.loc[
                        control_df["Fixed"].astype(str).str.upper() == "Y",
                        "PointID"
                    ].astype(str).tolist()
                )
                current_circuit_candidates = get_next_candidate_points(graph, current_circuit_path)
                current_circuit_type = classify_circuit_path(current_circuit_path, fixed_points) if len(current_circuit_path) >= 2 else ""

                if not errors:
                    success_message = (
                        f"Anchor elevation applied to {anchor_point_id}. Circuit elevations recalculated."
                    )

                active_stage = "adjustment"
                active_adjustment_tab = "circuit_summary"

            elif action == "apply_exclusions":
                if not raw_json or not leg_json or not decision_json:
                    errors.append("No analysis data found to apply exclusions.")
                elif not errors:
                    raw_df = parse_json_df(raw_json)
                    control_df = parse_json_df(control_json)
                    leg_df = parse_json_df(leg_json)
                    decision_df = parse_json_df(decision_json)

                    summary_df, updated_decision_df = recompute_after_exclusions(
                        decision_df, tolerance, selected_exclusions
                    )
                    cleaned_df = build_cleaned_leg_means(updated_decision_df, tolerance)
                    cleaned_df = sort_cleaned_df(cleaned_df)

                    raw_data = raw_df.to_dict(orient="records")
                    control_data = control_df.to_dict(orient="records")
                    leg_data = leg_df.to_dict(orient="records")
                    summary_data = summary_df.to_dict(orient="records")
                    decision_data = updated_decision_df.to_dict(orient="records")
                    cleaned_data = cleaned_df.to_dict(orient="records")

                    if not control_df.empty:
                        if adjustment_mode == "network":
                            (
                                adjusted_points_df,
                                observation_residuals_df,
                                control_checks_df,
                                connectivity_df,
                                sections_df,
                                adj_errors,
                                adj_warnings,
                            ) = run_network_pipeline(cleaned_df, control_df)

                            adjustment_messages.extend(adj_errors)
                            adjustment_messages.extend(adj_warnings)

                            adjusted_points_data = adjusted_points_df.to_dict(orient="records")
                            observation_residuals_data = observation_residuals_df.to_dict(orient="records")
                            control_checks_data = control_checks_df.to_dict(orient="records")
                            connectivity_data = connectivity_df.to_dict(orient="records")
                            sections_data = sections_df.to_dict(orient="records")
                        else:
                            (
                                circuit_summary_df,
                                circuit_legs_df,
                                circuit_elevations_df,
                                circuit_errors,
                                circuit_warnings,
                            ) = run_circuit_pipeline(saved_circuits, cleaned_df, control_df)

                            adjustment_messages.extend(circuit_errors)
                            adjustment_messages.extend(circuit_warnings)

                            circuit_summary_data = circuit_summary_df.to_dict(orient="records")
                            circuit_legs_data = circuit_legs_df.to_dict(orient="records")
                            circuit_elevations_data = circuit_elevations_df.to_dict(orient="records")

                            graph, _usable_cleaned = build_graph_from_cleaned_legs(cleaned_df)
                            available_start_points = get_all_available_points(graph)
                            fixed_points = set(
                                control_df.loc[
                                    control_df["Fixed"].astype(str).str.upper() == "Y",
                                    "PointID"
                                ].astype(str).tolist()
                            )
                            current_circuit_candidates = get_next_candidate_points(graph, current_circuit_path)
                            current_circuit_type = classify_circuit_path(current_circuit_path, fixed_points) if len(current_circuit_path) >= 2 else ""

                    success_message = "Exclusions applied successfully."
                    active_stage = "review"
                    active_tab = "decisions"

            elif action == "apply_uploaded_exclusions":
                if not raw_json or not leg_json or not decision_json:
                    errors.append("No analysis data found to apply uploaded exclusions.")
                elif not uploaded_state_file or uploaded_state_file.filename == "":
                    errors.append("Please select an exclusion state JSON file.")
                elif not errors:
                    raw_df = parse_json_df(raw_json)
                    control_df = parse_json_df(control_json)
                    leg_df = parse_json_df(leg_json)
                    decision_df = parse_json_df(decision_json)

                    selected_keys = parse_exclusion_state_file(uploaded_state_file)

                    summary_df, updated_decision_df = recompute_after_exclusions(
                        decision_df, tolerance, selected_keys
                    )
                    cleaned_df = build_cleaned_leg_means(updated_decision_df, tolerance)
                    cleaned_df = sort_cleaned_df(cleaned_df)

                    raw_data = raw_df.to_dict(orient="records")
                    control_data = control_df.to_dict(orient="records")
                    leg_data = leg_df.to_dict(orient="records")
                    summary_data = summary_df.to_dict(orient="records")
                    decision_data = updated_decision_df.to_dict(orient="records")
                    cleaned_data = cleaned_df.to_dict(orient="records")

                    if not control_df.empty:
                        if adjustment_mode == "network":
                            (
                                adjusted_points_df,
                                observation_residuals_df,
                                control_checks_df,
                                connectivity_df,
                                sections_df,
                                adj_errors,
                                adj_warnings,
                            ) = run_network_pipeline(cleaned_df, control_df)

                            adjustment_messages.extend(adj_errors)
                            adjustment_messages.extend(adj_warnings)

                            adjusted_points_data = adjusted_points_df.to_dict(orient="records")
                            observation_residuals_data = observation_residuals_df.to_dict(orient="records")
                            control_checks_data = control_checks_df.to_dict(orient="records")
                            connectivity_data = connectivity_df.to_dict(orient="records")
                            sections_data = sections_df.to_dict(orient="records")
                        else:
                            (
                                circuit_summary_df,
                                circuit_legs_df,
                                circuit_elevations_df,
                                circuit_errors,
                                circuit_warnings,
                            ) = run_circuit_pipeline(saved_circuits, cleaned_df, control_df)

                            adjustment_messages.extend(circuit_errors)
                            adjustment_messages.extend(circuit_warnings)

                            circuit_summary_data = circuit_summary_df.to_dict(orient="records")
                            circuit_legs_data = circuit_legs_df.to_dict(orient="records")
                            circuit_elevations_data = circuit_elevations_df.to_dict(orient="records")

                            graph, _usable_cleaned = build_graph_from_cleaned_legs(cleaned_df)
                            available_start_points = get_all_available_points(graph)
                            fixed_points = set(
                                control_df.loc[
                                    control_df["Fixed"].astype(str).str.upper() == "Y",
                                    "PointID"
                                ].astype(str).tolist()
                            )
                            current_circuit_candidates = get_next_candidate_points(graph, current_circuit_path)
                            current_circuit_type = classify_circuit_path(current_circuit_path, fixed_points) if len(current_circuit_path) >= 2 else ""

                    success_message = "Uploaded exclusion state applied successfully."
                    active_stage = "review"
                    active_tab = "decisions"

            else:
                if not uploaded_file or uploaded_file.filename == "":
                    errors.append("Please select an Excel file.")

                if not errors:
                    workbook = pd.ExcelFile(uploaded_file)

                    if "Field_Elevations" not in workbook.sheet_names:
                        errors.append("Could not find a sheet named 'Field_Elevations' in the Excel file.")

                    if not errors:
                        excel_data = pd.read_excel(workbook, sheet_name="Field_Elevations")
                        excel_data.columns = [str(c).strip() for c in excel_data.columns]

                        clean_df, validation_errors, validation_warnings = validate_field_data(excel_data)
                        errors.extend(validation_errors)
                        warnings.extend(validation_warnings)

                        if not errors:
                            raw_df = clean_df.copy()
                            leg_df = compute_legs(clean_df)
                            summary_df, decision_df = analyze_repeated_legs(leg_df, tolerance)
                            summary_df = sort_summary_df(summary_df)
                            decision_df = sort_decision_df(decision_df)

                            cleaned_df = build_cleaned_leg_means(decision_df, tolerance)
                            cleaned_df = sort_cleaned_df(cleaned_df)

                            raw_data = raw_df.to_dict(orient="records")
                            leg_data = leg_df.to_dict(orient="records")
                            summary_data = summary_df.to_dict(orient="records")
                            decision_data = decision_df.to_dict(orient="records")
                            cleaned_data = cleaned_df.to_dict(orient="records")

                            if "Control_Points" in workbook.sheet_names:
                                control_input_df = pd.read_excel(workbook, sheet_name="Control_Points")
                                control_input_df.columns = [str(c).strip() for c in control_input_df.columns]

                                control_df, control_errors, control_warnings = validate_control_points(control_input_df)

                                warnings.extend(control_warnings)

                                if control_errors:
                                    adjustment_messages.extend(control_errors)
                                    control_data = []
                                else:
                                    control_data = control_df.to_dict(orient="records")

                                    if adjustment_mode == "network":
                                        (
                                            adjusted_points_df,
                                            observation_residuals_df,
                                            control_checks_df,
                                            connectivity_df,
                                            sections_df,
                                            adj_errors,
                                            adj_warnings,
                                        ) = run_network_pipeline(cleaned_df, control_df)

                                        adjustment_messages.extend(adj_errors)
                                        adjustment_messages.extend(adj_warnings)

                                        adjusted_points_data = adjusted_points_df.to_dict(orient="records")
                                        observation_residuals_data = observation_residuals_df.to_dict(orient="records")
                                        control_checks_data = control_checks_df.to_dict(orient="records")
                                        connectivity_data = connectivity_df.to_dict(orient="records")
                                        sections_data = sections_df.to_dict(orient="records")
                                    else:
                                        graph, _usable_cleaned = build_graph_from_cleaned_legs(cleaned_df)
                                        available_start_points = get_all_available_points(graph)

                                        (
                                            circuit_summary_df,
                                            circuit_legs_df,
                                            circuit_elevations_df,
                                            circuit_errors,
                                            circuit_warnings,
                                        ) = run_circuit_pipeline(saved_circuits, cleaned_df, control_df)

                                        adjustment_messages.extend(circuit_errors)
                                        adjustment_messages.extend(circuit_warnings)

                                        circuit_summary_data = circuit_summary_df.to_dict(orient="records")
                                        circuit_legs_data = circuit_legs_df.to_dict(orient="records")
                                        circuit_elevations_data = circuit_elevations_df.to_dict(orient="records")
                            else:
                                adjustment_messages.append(
                                    "Control_Points sheet was not found. Adjustment stage is unavailable."
                                )
                                control_data = []

                            success_message = "File loaded and analysis completed successfully."

        except Exception as exc:
            errors.append(f"Unexpected error while processing the file: {exc}")

    return render_template(
        "index.html",
        raw_data=raw_data,
        control_data=control_data,
        leg_data=leg_data,
        summary_data=summary_data,
        decision_data=decision_data,
        cleaned_data=cleaned_data,
        adjusted_points_data=adjusted_points_data,
        observation_residuals_data=observation_residuals_data,
        control_checks_data=control_checks_data,
        connectivity_data=connectivity_data,
        sections_data=sections_data,
        circuit_summary_data=circuit_summary_data,
        circuit_legs_data=circuit_legs_data,
        circuit_elevations_data=circuit_elevations_data,
        current_circuit_path=current_circuit_path,
        current_circuit_candidates=current_circuit_candidates,
        available_start_points=available_start_points,
        current_circuit_type=current_circuit_type,
        current_circuit_message=current_circuit_message,
        saved_circuits=saved_circuits,
        saved_circuits_json=json.dumps(saved_circuits),
        unanchored_circuit_options=build_unanchored_circuit_options(
            saved_circuits,
            pd.DataFrame(control_data or []),
        ),
        unassigned_points=build_unassigned_points(
            available_start_points, current_circuit_path, saved_circuits
        ),
        tolerance=tolerance,
        residual_threshold=residual_threshold,
        errors=errors,
        warnings=warnings,
        adjustment_messages=adjustment_messages,
        success_message=success_message,
        active_stage=active_stage,
        active_tab=active_tab,
        active_adjustment_tab=active_adjustment_tab,
        adjustment_mode=adjustment_mode,
    )


if __name__ == "__main__":
    app.run(debug=True)
