from io import BytesIO
import pandas as pd


def format_export_raw_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df
    work = raw_df.copy().rename(columns={
        "RunID": "Run ID",
        "Sequence": "Sequence",
        "PointID": "Point ID",
        "Raw_Elevation": "Raw Elevation",
    })
    return work[["Run ID", "Sequence", "Point ID", "Raw Elevation"]]


def format_export_control_df(control_df: pd.DataFrame) -> pd.DataFrame:
    if control_df.empty:
        return control_df
    work = control_df.copy().rename(columns={
        "PointID": "Point ID",
        "Elevation": "Elevation",
        "Fixed": "Fixed",
    })
    return work[["Point ID", "Elevation", "Fixed"]]


def format_export_computed_legs_df(leg_df: pd.DataFrame) -> pd.DataFrame:
    if leg_df.empty:
        return leg_df
    work = leg_df.copy().rename(columns={
        "FromPoint": "From Point",
        "ToPoint": "To Point",
        "RunID": "Run ID",
        "FromSeq": "From Sequence",
        "ToSeq": "To Sequence",
        "LegID": "Leg ID",
        "RawDiff": "Raw Delta Z",
        "NormalizedDiff": "Normalized Delta Z",
    })
    return work[
        ["From Point", "To Point", "Run ID", "From Sequence", "To Sequence", "Leg ID", "Raw Delta Z", "Normalized Delta Z"]
    ]


def format_export_summary_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    work = summary_df.copy().rename(columns={
        "Leg_ID": "Leg ID",
        "Observation_Count": "Observation Count",
        "Reference_Value": "Reference Value (Median)",
        "Maximum_Residual": "Maximum Residual",
        "Status": "Status",
    })
    return work[
        ["Leg ID", "Observation Count", "Reference Value (Median)", "Maximum Residual", "Status"]
    ]


def format_export_decisions_df(decision_df: pd.DataFrame) -> pd.DataFrame:
    if decision_df.empty:
        return decision_df
    work = decision_df.copy().rename(columns={
        "Observation_ID": "Observation ID",
        "From_Point": "From Point",
        "To_Point": "To Point",
        "Leg_ID": "Leg ID",
        "Run_ID": "Run ID",
        "Normalized_Delta_Z": "Normalized Delta Z",
        "Reference_Value": "Reference Value (Median)",
        "Residual_From_Reference": "Residual From Reference",
        "Decision": "Decision",
        "User_Excluded": "User Excluded",
    })
    columns = [
        "From Point",
        "To Point",
        "Leg ID",
        "Run ID",
        "Normalized Delta Z",
        "Reference Value (Median)",
        "Residual From Reference",
        "Decision",
        "User Excluded",
    ]
    if "Observation ID" in work.columns:
        columns.insert(0, "Observation ID")
    return work[columns]


def format_export_cleaned_df(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    if cleaned_df.empty:
        return cleaned_df
    work = cleaned_df.copy().rename(columns={
        "From_Point": "From Point",
        "To_Point": "To Point",
        "Leg_ID": "Leg ID",
        "Accepted_Observation_Count": "Accepted Observation Count",
        "Final_Normalized_Delta_Z": "Final Normalized Delta Z",
        "Standard_Deviation": "Standard Deviation",
        "Status": "Status",
    })
    return work[
        ["From Point", "To Point", "Leg ID", "Accepted Observation Count", "Final Normalized Delta Z", "Standard Deviation", "Status"]
    ]


def format_export_adjusted_points_df(adjusted_points_df: pd.DataFrame) -> pd.DataFrame:
    if adjusted_points_df.empty:
        return adjusted_points_df
    work = adjusted_points_df.copy().rename(columns={
        "Point_ID": "Point ID",
        "Adjusted_Elevation": "Adjusted Elevation",
        "Sigma": "Sigma",
        "Fixed": "Fixed",
        "Control_Elevation": "Control Elevation",
        "Difference_To_Control": "Difference To Control",
    })
    return work[
        ["Point ID", "Adjusted Elevation", "Sigma", "Fixed", "Control Elevation", "Difference To Control"]
    ]


def format_export_observation_residuals_df(residuals_df: pd.DataFrame) -> pd.DataFrame:
    if residuals_df.empty:
        return residuals_df
    work = residuals_df.copy().rename(columns={
        "From_Point": "From Point",
        "To_Point": "To Point",
        "Leg_ID": "Leg ID",
        "Final_Normalized_Delta_Z": "Final Normalized Delta Z",
        "Adjusted_Delta_Z": "Adjusted Delta Z",
        "Residual": "Residual",
        "Weight": "Weight",
        "Used_In_Adjustment": "Used In Adjustment",
    })
    return work[
        ["From Point", "To Point", "Leg ID", "Final Normalized Delta Z", "Adjusted Delta Z", "Residual", "Weight", "Used In Adjustment"]
    ]


def format_export_control_checks_df(control_checks_df: pd.DataFrame) -> pd.DataFrame:
    if control_checks_df.empty:
        return control_checks_df
    work = control_checks_df.copy().rename(columns={
        "Point_ID": "Point ID",
        "Control_Elevation": "Control Elevation",
        "Adjusted_Elevation": "Adjusted Elevation",
        "Fixed": "Fixed",
        "Difference": "Difference",
    })
    return work[
        ["Point ID", "Control Elevation", "Adjusted Elevation", "Fixed", "Difference"]
    ]


def format_export_connectivity_df(connectivity_df: pd.DataFrame) -> pd.DataFrame:
    if connectivity_df.empty:
        return connectivity_df
    work = connectivity_df.copy().rename(columns={
        "Component_ID": "Component ID",
        "Point_Count": "Point Count",
        "Observation_Count": "Observation Count",
        "Unknown_Count": "Unknown Count",
        "Degrees_Of_Freedom": "Degrees Of Freedom",
        "Fixed_Point_Count": "Fixed Point Count",
        "Fixed_Points": "Fixed Points",
        "Status": "Status",
        "Redundancy_Status": "Redundancy Status",
        "Points": "Points",
    })
    return work[
        ["Component ID", "Point Count", "Observation Count", "Unknown Count", "Degrees Of Freedom", "Fixed Point Count", "Fixed Points", "Status", "Redundancy Status", "Points"]
    ]


def format_export_sections_df(sections_df: pd.DataFrame) -> pd.DataFrame:
    if sections_df.empty:
        return sections_df
    work = sections_df.copy().rename(columns={
        "Section_ID": "Section ID",
        "Component_ID": "Component ID",
        "Start_Point": "Start Point",
        "End_Point": "End Point",
        "Start_Type": "Start Type",
        "End_Type": "End Type",
        "Section_Type": "Section Type",
        "Leg_Count": "Leg Count",
        "Point_Count": "Point Count",
        "Points_In_Order": "Points In Order",
        "Legs_In_Order": "Legs In Order",
        "Adjustable": "Adjustable",
    })
    return work[
        ["Section ID", "Component ID", "Start Point", "End Point", "Start Type", "End Type", "Section Type", "Leg Count", "Point Count", "Adjustable", "Points In Order", "Legs In Order"]
    ]


def format_export_circuit_summary_df(circuit_summary_df: pd.DataFrame) -> pd.DataFrame:
    if circuit_summary_df.empty:
        return circuit_summary_df
    work = circuit_summary_df.copy().rename(columns={
        "Circuit_ID": "Circuit ID",
        "Circuit_Type": "Circuit Type",
        "Point_Count": "Point Count",
        "Leg_Count": "Leg Count",
        "Start_Point": "Start Point",
        "End_Point": "End Point",
        "Observed_Total_Delta_Z": "Observed Total Delta Z",
        "Control_Delta_Z": "Control Delta Z",
        "Closure_Error": "Closure Error",
        "Correction_Per_Leg": "Correction Per Leg",
        "Status": "Status",
    })
    return work[
        ["Circuit ID", "Circuit Type", "Point Count", "Leg Count", "Start Point", "End Point", "Observed Total Delta Z", "Control Delta Z", "Closure Error", "Correction Per Leg", "Status"]
    ]


def format_export_circuit_legs_df(circuit_legs_df: pd.DataFrame) -> pd.DataFrame:
    if circuit_legs_df.empty:
        return circuit_legs_df
    work = circuit_legs_df.copy().rename(columns={
        "Circuit_ID": "Circuit ID",
        "From_Point": "From Point",
        "To_Point": "To Point",
        "Leg_ID": "Leg ID",
        "Observed_Delta_Z": "Observed Delta Z",
        "Correction": "Proration Correction",
        "Corrected_Delta_Z": "Corrected Delta Z",
    })
    return work[
        ["Circuit ID", "From Point", "To Point", "Leg ID", "Observed Delta Z", "Proration Correction", "Corrected Delta Z"]
    ]


def format_export_circuit_elevations_df(circuit_elevations_df: pd.DataFrame) -> pd.DataFrame:
    if circuit_elevations_df.empty:
        return circuit_elevations_df
    work = circuit_elevations_df.copy().rename(columns={
        "Circuit_ID": "Circuit ID",
        "Point_ID": "Point ID",
        "Elevation": "Elevation",
    })
    return work[["Circuit ID", "Point ID", "Elevation"]]


def export_analysis_workbook(
    raw_df, control_df, leg_df, summary_df, decision_df, cleaned_df,
    adjusted_points_df, observation_residuals_df, control_checks_df,
    connectivity_df, sections_df,
    circuit_summary_df, circuit_legs_df, circuit_elevations_df,
):
    output = BytesIO()

    raw_export = format_export_raw_df(raw_df)
    control_export = format_export_control_df(control_df)
    legs_export = format_export_computed_legs_df(leg_df)
    summary_export = format_export_summary_df(summary_df)
    decisions_export = format_export_decisions_df(decision_df)
    cleaned_export = format_export_cleaned_df(cleaned_df)
    adjusted_points_export = format_export_adjusted_points_df(adjusted_points_df)
    observation_residuals_export = format_export_observation_residuals_df(observation_residuals_df)
    control_checks_export = format_export_control_checks_df(control_checks_df)
    connectivity_export = format_export_connectivity_df(connectivity_df)
    sections_export = format_export_sections_df(sections_df)
    circuit_summary_export = format_export_circuit_summary_df(circuit_summary_df)
    circuit_legs_export = format_export_circuit_legs_df(circuit_legs_df)
    circuit_elevations_export = format_export_circuit_elevations_df(circuit_elevations_df)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if not raw_export.empty:
            raw_export.to_excel(writer, sheet_name="Raw Field Observations", index=False)
        if not control_export.empty:
            control_export.to_excel(writer, sheet_name="Control Points", index=False)
        if not legs_export.empty:
            legs_export.to_excel(writer, sheet_name="Computed Legs", index=False)
        if not summary_export.empty:
            summary_export.to_excel(writer, sheet_name="Repeated Leg Analysis", index=False)
        if not decisions_export.empty:
            decisions_export.to_excel(writer, sheet_name="Observation Decisions", index=False)
        if not cleaned_export.empty:
            cleaned_export.to_excel(writer, sheet_name="Cleaned Leg Means", index=False)
        if not connectivity_export.empty:
            connectivity_export.to_excel(writer, sheet_name="Connectivity Report", index=False)
        if not sections_export.empty:
            sections_export.to_excel(writer, sheet_name="Network Sections", index=False)
        if not adjusted_points_export.empty:
            adjusted_points_export.to_excel(writer, sheet_name="Adjusted Elevations", index=False)
        if not observation_residuals_export.empty:
            observation_residuals_export.to_excel(writer, sheet_name="Observation Residuals", index=False)
        if not control_checks_export.empty:
            control_checks_export.to_excel(writer, sheet_name="Control Point Checks", index=False)
        if not circuit_summary_export.empty:
            circuit_summary_export.to_excel(writer, sheet_name="Circuit Summary", index=False)
        if not circuit_legs_export.empty:
            circuit_legs_export.to_excel(writer, sheet_name="Circuit Legs", index=False)
        if not circuit_elevations_export.empty:
            circuit_elevations_export.to_excel(writer, sheet_name="Circuit Elevations", index=False)

    output.seek(0)
    return output
