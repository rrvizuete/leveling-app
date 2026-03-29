import pandas as pd


CONTROL_REQUIRED_COLUMNS = ["PointID", "Elevation", "Fixed"]


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


def update_control_fixed_flags(control_df: pd.DataFrame, point_ids: list[str], fixed_values: list[str]):
    work = control_df.copy()

    if len(point_ids) != len(fixed_values):
        return work, ["Control point update could not be applied because the submitted data is inconsistent."]

    update_map = {
        str(pid): str(fixed).strip().upper()
        for pid, fixed in zip(point_ids, fixed_values)
    }

    errors = []

    for value in update_map.values():
        if value not in {"Y", "N"}:
            errors.append("Fixed values must be Y or N.")
            return work, errors

    work["PointID"] = work["PointID"].astype(str)
    work["Fixed"] = work.apply(
        lambda row: update_map.get(str(row["PointID"]), str(row["Fixed"]).strip().upper()),
        axis=1
    )

    return work, errors