import pandas as pd


REQUIRED_COLUMNS = ["RunID", "Sequence", "PointID", "Raw_Elevation"]


def clean_text_identifier(value):
    """
    Ensures identifiers like PointID and RunID are treated as text.
    Removes Excel float artifacts like 121.0 -> "121".
    """

    if pd.isna(value):
        return ""

    # If Excel stored it as float like 121.0
    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value).strip()


def validate_field_data(df: pd.DataFrame):

    errors = []
    warnings = []

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        errors.append(
            f"Missing required columns in Field_Elevations: {', '.join(missing)}"
        )
        return pd.DataFrame(), errors, warnings

    if df.empty:
        errors.append("Field_Elevations sheet is empty.")
        return pd.DataFrame(), errors, warnings

    work = df.copy()

    work = work[REQUIRED_COLUMNS].copy()

    # -------------------------------------------------
    # Remove completely blank rows
    # -------------------------------------------------

    blank_mask = work.isna().all(axis=1)
    blank_rows = work.index[blank_mask].tolist()

    if blank_rows:
        warnings.append(
            f"Ignored completely blank rows at Excel rows: "
            f"{', '.join(str(i + 2) for i in blank_rows)}"
        )

    work = work.loc[~blank_mask].copy()

    if work.empty:
        errors.append(
            "Field_Elevations contains no usable rows after removing blank rows."
        )
        return pd.DataFrame(), errors, warnings

    # -------------------------------------------------
    # Detect partially blank rows
    # -------------------------------------------------

    partial_blank_mask = work.isna().any(axis=1)
    partial_rows = work.index[partial_blank_mask].tolist()

    if partial_rows:
        warnings.append(
            f"Discarded rows with blank required cells at Excel rows: "
            f"{', '.join(str(i + 2) for i in partial_rows)}"
        )

    work = work.loc[~partial_blank_mask].copy()

    if work.empty:
        errors.append(
            "Field_Elevations contains no usable rows after discarding incomplete rows."
        )
        return pd.DataFrame(), errors, warnings

    # -------------------------------------------------
    # Clean identifiers (IMPORTANT FIX)
    # -------------------------------------------------

    work["RunID"] = work["RunID"].apply(clean_text_identifier)
    work["PointID"] = work["PointID"].apply(clean_text_identifier)

    # Remove empty identifiers
    empty_mask = (work["RunID"] == "") | (work["PointID"] == "")

    if empty_mask.any():
        rows = work.index[empty_mask].tolist()

        warnings.append(
            f"Discarded rows with empty identifiers at Excel rows: "
            f"{', '.join(str(i + 2) for i in rows)}"
        )

    work = work.loc[~empty_mask].copy()

    # -------------------------------------------------
    # Numeric validation
    # -------------------------------------------------

    sequence_numeric = pd.to_numeric(work["Sequence"], errors="coerce")
    elevation_numeric = pd.to_numeric(work["Raw_Elevation"], errors="coerce")

    bad_seq = sequence_numeric.isna()
    bad_elev = elevation_numeric.isna()

    if bad_seq.any():
        rows = work.index[bad_seq].tolist()

        warnings.append(
            f"Discarded rows with non-numeric Sequence values at Excel rows: "
            f"{', '.join(str(i + 2) for i in rows)}"
        )

    if bad_elev.any():
        rows = work.index[bad_elev].tolist()

        warnings.append(
            f"Discarded rows with non-numeric Raw_Elevation values at Excel rows: "
            f"{', '.join(str(i + 2) for i in rows)}"
        )

    valid_mask = ~(bad_seq | bad_elev)

    work = work.loc[valid_mask].copy()
    sequence_numeric = sequence_numeric.loc[valid_mask]
    elevation_numeric = elevation_numeric.loc[valid_mask]

    if work.empty:
        errors.append(
            "Field_Elevations contains no usable rows after discarding non-numeric values."
        )
        return pd.DataFrame(), errors, warnings

    work["Sequence"] = sequence_numeric.astype(int)
    work["Raw_Elevation"] = elevation_numeric.astype(float)

    # -------------------------------------------------
    # Detect duplicate sequence numbers in run
    # -------------------------------------------------

    dupes = work.duplicated(subset=["RunID", "Sequence"], keep=False)

    if dupes.any():
        rows = work.index[dupes].tolist()

        errors.append(
            "Duplicate Sequence values found within the same RunID at Excel rows: "
            + ", ".join(str(i + 2) for i in rows)
        )

    # -------------------------------------------------
    # Check minimum observations per run
    # -------------------------------------------------

    counts = work.groupby("RunID").size()

    short_runs = counts[counts < 2]

    if not short_runs.empty:
        runs = ", ".join(short_runs.index.astype(str).tolist())

        errors.append(f"These runs have fewer than 2 observations: {runs}")

    return work, errors, warnings


def compute_legs(df: pd.DataFrame):

    work = df.copy()

    work = work.sort_values(["RunID", "Sequence"]).reset_index(drop=True)

    leg_rows = []

    for run_id, group in work.groupby("RunID", sort=False):

        group = group.sort_values("Sequence").reset_index(drop=True)

        for i in range(len(group) - 1):

            from_row = group.iloc[i]
            to_row = group.iloc[i + 1]

            from_point = from_row["PointID"]
            to_point = to_row["PointID"]

            raw_diff = float(to_row["Raw_Elevation"]) - float(
                from_row["Raw_Elevation"]
            )

            point_a, point_b = sorted([from_point, to_point])

            leg_id = f"{point_a}|{point_b}"

            if from_point == point_a and to_point == point_b:
                normalized_diff = raw_diff
            else:
                normalized_diff = -raw_diff

            leg_rows.append(
                {
                    "RunID": run_id,
                    "FromSeq": int(from_row["Sequence"]),
                    "ToSeq": int(to_row["Sequence"]),
                    "FromPoint": from_point,
                    "ToPoint": to_point,
                    "LegID": leg_id,
                    "RawDiff": round(raw_diff, 4),
                    "NormalizedDiff": round(normalized_diff, 4),
                }
            )

    return pd.DataFrame(leg_rows)