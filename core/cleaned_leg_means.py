import pandas as pd


def build_cleaned_leg_means(decision_df: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """
    Build one cleaned row per leg using observations that are NOT excluded by the user.

    Final Normalized Delta Z = arithmetic mean of active observations.
    Standard Deviation = sample standard deviation when count > 1.

    Status rules:
    - All Observations Excluded
    - Single Observation
    - Review Required
    - Ready
    """
    if decision_df.empty:
        return pd.DataFrame(
            columns=[
                "Leg_ID",
                "From_Point",
                "To_Point",
                "Accepted_Observation_Count",
                "Final_Normalized_Delta_Z",
                "Standard_Deviation",
                "Status",
            ]
        )

    rows = []

    for leg_id, group in decision_df.groupby("Leg_ID", sort=False):
        active_group = group.loc[~group["User_Excluded"]].copy()

        if "|" in str(leg_id):
            from_point, to_point = str(leg_id).split("|", 1)
        else:
            from_point = str(leg_id)
            to_point = ""

        if active_group.empty:
            rows.append(
                {
                    "Leg_ID": leg_id,
                    "From_Point": from_point,
                    "To_Point": to_point,
                    "Accepted_Observation_Count": 0,
                    "Final_Normalized_Delta_Z": "",
                    "Standard_Deviation": "",
                    "Status": "All Observations Excluded",
                }
            )
            continue

        diffs = active_group["Normalized_Delta_Z"].astype(float)
        count = len(diffs)
        final_mean = float(diffs.mean())

        if count > 1:
            std_dev = float(diffs.std(ddof=1))
        else:
            std_dev = ""

        active_decisions = set(active_group["Decision"].astype(str).tolist())

        if count == 1:
            status = "Single Observation"
        elif "Pending Review" in active_decisions or "Suspect" in active_decisions:
            status = "Review Required"
        else:
            status = "Ready"

        rows.append(
            {
                "Leg_ID": leg_id,
                "From_Point": from_point,
                "To_Point": to_point,
                "Accepted_Observation_Count": count,
                "Final_Normalized_Delta_Z": round(final_mean, 4),
                "Standard_Deviation": round(std_dev, 4) if std_dev != "" else "",
                "Status": status,
            }
        )

    return pd.DataFrame(rows)