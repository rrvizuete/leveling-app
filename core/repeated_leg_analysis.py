import pandas as pd


def analyze_repeated_legs(legs_df: pd.DataFrame, tolerance: float):
    grouped = legs_df.groupby("LegID")

    summary_rows = []
    decision_rows = []

    for leg_id, group in grouped:
        diffs = group["NormalizedDiff"].tolist()
        observation_count = len(diffs)

        reference_value = pd.Series(diffs).median()
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
            diff = float(row["NormalizedDiff"])
            residual = abs(diff - reference_value)

            if observation_count == 1:
                decision = "Single Observation"
            elif observation_count == 2 and abs(diffs[0] - diffs[1]) > tolerance:
                decision = "Pending Review"
            else:
                decision = "Accepted" if residual <= tolerance else "Suspect"

            decision_rows.append(
                {
                    "Leg_ID": leg_id,
                    "Run_ID": row["RunID"],
                    "From_Point": row["FromPoint"],
                    "To_Point": row["ToPoint"],
                    "Normalized_Delta_Z": round(diff, 4),
                    "Reference_Value": round(reference_value, 4),
                    "Residual_From_Reference": round(residual, 4),
                    "Decision": decision,
                    "User_Excluded": False,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    decisions_df = pd.DataFrame(decision_rows)

    return summary_df, decisions_df