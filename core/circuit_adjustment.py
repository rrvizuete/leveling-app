import pandas as pd


def classify_circuit_from_points(point_ids: list[str], fixed_points: set[str]) -> str:
    if not point_ids or len(point_ids) < 2:
        return "Invalid"

    start_fixed = point_ids[0] in fixed_points
    end_fixed = point_ids[-1] in fixed_points

    if start_fixed and end_fixed:
        return "Fixed to Fixed"
    if start_fixed or end_fixed:
        return "Fixed to Free"
    return "Unanchored"


def compute_circuit_adjustment(saved_circuits: list[dict], control_df: pd.DataFrame):
    errors = []
    warnings = []

    circuit_summary_rows = []
    circuit_leg_rows = []
    circuit_elevation_rows = []

    if not saved_circuits:
        warnings.append("No saved circuits available.")
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            errors,
            warnings,
        )

    control_map = {
        str(row["PointID"]): float(row["Elevation"])
        for _, row in control_df.iterrows()
    }

    fixed_points = set(
        control_df.loc[
            control_df["Fixed"].astype(str).str.upper() == "Y",
            "PointID"
        ].astype(str).tolist()
    )

    for circuit in saved_circuits:
        circuit_id = circuit["Circuit_ID"]
        path = [str(p) for p in circuit["Path"]]
        legs = circuit["Legs"]

        if len(path) < 2 or not legs:
            circuit_summary_rows.append(
                {
                    "Circuit_ID": circuit_id,
                    "Circuit_Type": "Invalid",
                    "Point_Count": len(path),
                    "Leg_Count": len(legs),
                    "Start_Point": path[0] if path else "",
                    "End_Point": path[-1] if path else "",
                    "Observed_Total_Delta_Z": "",
                    "Control_Delta_Z": "",
                    "Closure_Error": "",
                    "Correction_Per_Leg": "",
                    "Status": "Invalid circuit.",
                }
            )
            continue

        circuit_type = classify_circuit_from_points(path, fixed_points)
        start_point = path[0]
        end_point = path[-1]
        leg_count = len(legs)
        point_count = len(path)

        observed_values = []
        missing_leg = False

        for leg in legs:
            dz = leg.get("Observed_Delta_Z", "")
            if dz == "":
                missing_leg = True
                break
            observed_values.append(float(dz))

        if missing_leg:
            circuit_summary_rows.append(
                {
                    "Circuit_ID": circuit_id,
                    "Circuit_Type": circuit_type,
                    "Point_Count": point_count,
                    "Leg_Count": leg_count,
                    "Start_Point": start_point,
                    "End_Point": end_point,
                    "Observed_Total_Delta_Z": "",
                    "Control_Delta_Z": "",
                    "Closure_Error": "",
                    "Correction_Per_Leg": "",
                    "Status": "One or more legs are missing from cleaned leg means.",
                }
            )
            continue

        observed_total = sum(observed_values)

        if circuit_type == "Fixed to Fixed":
            if start_point not in control_map or end_point not in control_map:
                circuit_summary_rows.append(
                    {
                        "Circuit_ID": circuit_id,
                        "Circuit_Type": circuit_type,
                        "Point_Count": point_count,
                        "Leg_Count": leg_count,
                        "Start_Point": start_point,
                        "End_Point": end_point,
                        "Observed_Total_Delta_Z": round(observed_total, 4),
                        "Control_Delta_Z": "",
                        "Closure_Error": "",
                        "Correction_Per_Leg": "",
                        "Status": "Start or end fixed point elevation not found.",
                    }
                )
                continue

            control_delta_z = control_map[end_point] - control_map[start_point]
            closure_error = control_delta_z - observed_total
            correction_per_leg = closure_error / leg_count if leg_count > 0 else 0.0

            current_elevation = control_map[start_point]

            circuit_elevation_rows.append(
                {
                    "Circuit_ID": circuit_id,
                    "Point_ID": start_point,
                    "Elevation": round(current_elevation, 4),
                }
            )

            for leg in legs:
                correction = correction_per_leg
                corrected_dz = float(leg["Observed_Delta_Z"]) + correction
                current_elevation += corrected_dz

                circuit_leg_rows.append(
                    {
                        "Circuit_ID": circuit_id,
                        "From_Point": leg["From_Point"],
                        "To_Point": leg["To_Point"],
                        "Leg_ID": leg["Leg_ID"],
                        "Observed_Delta_Z": round(float(leg["Observed_Delta_Z"]), 4),
                        "Correction": round(correction, 4),
                        "Corrected_Delta_Z": round(corrected_dz, 4),
                    }
                )

                circuit_elevation_rows.append(
                    {
                        "Circuit_ID": circuit_id,
                        "Point_ID": leg["To_Point"],
                        "Elevation": round(current_elevation, 4),
                    }
                )

            circuit_summary_rows.append(
                {
                    "Circuit_ID": circuit_id,
                    "Circuit_Type": circuit_type,
                    "Point_Count": point_count,
                    "Leg_Count": leg_count,
                    "Start_Point": start_point,
                    "End_Point": end_point,
                    "Observed_Total_Delta_Z": round(observed_total, 4),
                    "Control_Delta_Z": round(control_delta_z, 4),
                    "Closure_Error": round(closure_error, 4),
                    "Correction_Per_Leg": round(correction_per_leg, 4),
                    "Status": "Adjusted by closure proration.",
                }
            )

        elif circuit_type == "Fixed to Free":
            anchor_point = start_point if start_point in fixed_points else end_point

            if anchor_point not in control_map:
                circuit_summary_rows.append(
                    {
                        "Circuit_ID": circuit_id,
                        "Circuit_Type": circuit_type,
                        "Point_Count": point_count,
                        "Leg_Count": leg_count,
                        "Start_Point": start_point,
                        "End_Point": end_point,
                        "Observed_Total_Delta_Z": round(observed_total, 4),
                        "Control_Delta_Z": "",
                        "Closure_Error": "",
                        "Correction_Per_Leg": "",
                        "Status": "Anchor fixed point elevation not found.",
                    }
                )
                continue

            if anchor_point == start_point:
                ordered_legs = legs
                ordered_path = path
                current_elevation = control_map[start_point]
            else:
                ordered_path = list(reversed(path))
                reversed_legs = []
                for leg in reversed(legs):
                    reversed_legs.append(
                        {
                            "From_Point": leg["To_Point"],
                            "To_Point": leg["From_Point"],
                            "Leg_ID": leg["Leg_ID"],
                            "Observed_Delta_Z": -float(leg["Observed_Delta_Z"]),
                        }
                    )
                ordered_legs = reversed_legs
                current_elevation = control_map[end_point]

            circuit_elevation_rows.append(
                {
                    "Circuit_ID": circuit_id,
                    "Point_ID": ordered_path[0],
                    "Elevation": round(current_elevation, 4),
                }
            )

            for leg in ordered_legs:
                corrected_dz = float(leg["Observed_Delta_Z"])
                current_elevation += corrected_dz

                circuit_leg_rows.append(
                    {
                        "Circuit_ID": circuit_id,
                        "From_Point": leg["From_Point"],
                        "To_Point": leg["To_Point"],
                        "Leg_ID": leg["Leg_ID"],
                        "Observed_Delta_Z": round(float(leg["Observed_Delta_Z"]), 4),
                        "Correction": "",
                        "Corrected_Delta_Z": round(corrected_dz, 4),
                    }
                )

                circuit_elevation_rows.append(
                    {
                        "Circuit_ID": circuit_id,
                        "Point_ID": leg["To_Point"],
                        "Elevation": round(current_elevation, 4),
                    }
                )

            circuit_summary_rows.append(
                {
                    "Circuit_ID": circuit_id,
                    "Circuit_Type": circuit_type,
                    "Point_Count": point_count,
                    "Leg_Count": leg_count,
                    "Start_Point": start_point,
                    "End_Point": end_point,
                    "Observed_Total_Delta_Z": round(observed_total, 4),
                    "Control_Delta_Z": "",
                    "Closure_Error": "",
                    "Correction_Per_Leg": "",
                    "Status": "Open branch from fixed point. No closure applied.",
                }
            )

        else:
            warnings.append(
                f"{circuit_id} is a free-to-free circuit. Provide a known elevation for at least one point in this circuit (set it as Fixed) so elevations can be computed."
            )
            circuit_summary_rows.append(
                {
                    "Circuit_ID": circuit_id,
                    "Circuit_Type": circuit_type,
                    "Point_Count": point_count,
                    "Leg_Count": leg_count,
                    "Start_Point": start_point,
                    "End_Point": end_point,
                    "Observed_Total_Delta_Z": round(observed_total, 4),
                    "Control_Delta_Z": "",
                    "Closure_Error": "",
                    "Correction_Per_Leg": "",
                    "Status": "Unanchored free-to-free circuit. Enter elevation on at least one point to anchor adjustment.",
                }
            )

    return (
        pd.DataFrame(circuit_summary_rows),
        pd.DataFrame(circuit_leg_rows),
        pd.DataFrame(circuit_elevation_rows),
        errors,
        warnings,
    )
