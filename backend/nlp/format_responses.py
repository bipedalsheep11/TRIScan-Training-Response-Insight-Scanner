# backend/nlp/format_responses.py
# ─────────────────────────────────────────────────────────────────
# Converts a labeled DataFrame into LLM-readable strings.
# Produces two outputs per cluster:
#   1. formatted_ratings  — average Likert scores as a readable block
#   2. formatted_responses — respondent text in pseudo-CSV format
# ─────────────────────────────────────────────────────────────────

import pandas as pd


def generate_formatted_responses(
    labeled_df:  pd.DataFrame,
    cluster_id:  int,
    likert_cols: list[str],
    text_cols:   list[str],
) -> list[str]:
    """
    Format one cluster's data into two LLM-readable strings.

    Parameters
    ----------
    labeled_df  : pd.DataFrame — full survey data with 'cluster' column
    cluster_id  : int — which cluster to format (0-indexed integer)
    likert_cols : list[str] — column names for Likert rating questions
    text_cols   : list[str] — column names for free-text response questions

    Returns
    -------
    list of two strings: [formatted_ratings, formatted_responses]
      formatted_ratings   — average rating per Likert question
      formatted_responses — CSV-like block of respondent text answers
    """
    # ── Average Ratings ──────────────────────────────────────────
    if likert_cols:
        # .groupby('cluster') groups the rows by cluster assignment.
        # [likert_cols] selects only the Likert columns.
        # .mean() computes the average value in each column for each cluster.
        # .round(2) rounds to 2 decimal places for readability.
        avg_ratings = labeled_df.groupby("cluster")[likert_cols].mean().round(2)

        # .iloc[cluster_id, :] selects the row corresponding to this cluster.
        # .items() returns (column_name, value) pairs so we can format them.
        rating_lines = "\n ".join(
            f"{col}: {round(val, 1)}/5"
            for col, val in avg_ratings.loc[cluster_id].items()
        )
    else:
        rating_lines = "(no Likert columns detected)"

    formatted_ratings = f"Cluster: {cluster_id}\n{'=' * 30}\n{rating_lines}"

    # ── Text Responses ───────────────────────────────────────────
    # Filter to only the rows belonging to this cluster
    cluster_rows     = labeled_df.loc[labeled_df["cluster"] == cluster_id]
    sample_responses = cluster_rows[text_cols].to_numpy() if text_cols else []
    respondent_index = cluster_rows.index.to_numpy()

    if text_cols and len(sample_responses) > 0:
        # Header line: "Respondent, Question1, Question2, ..."
        header = "Respondent, " + ", ".join(text_cols)

        # Each data line: "row_number, answer1, answer2, ..."
        # We use the original DataFrame index so IDs are stable
        data_lines = "\n".join(
            f"{idx + 1}, " + ", ".join(str(t) for t in text_row)
            for idx, text_row in zip(respondent_index, sample_responses)
        )
        formatted_responses = f"Cluster: {cluster_id}\n{'=' * 30}\n{header}\n{data_lines}"
    else:
        formatted_responses = f"Cluster: {cluster_id}\n{'=' * 30}\n(no text columns detected)"

    return [formatted_ratings, formatted_responses]


def get_all_clusters_table(
    labeled_df:  pd.DataFrame,
    best_k:      int,
    likert_cols: list[str],
    text_cols:   list[str],
) -> list[str]:
    """
    Concatenate formatted ratings and responses for ALL clusters
    into two combined strings — used as context when labelling
    individual clusters (so the LLM can compare across clusters).

    Parameters
    ----------
    labeled_df  : pd.DataFrame — labeled survey data
    best_k      : int — number of clusters
    likert_cols : list[str] — Likert column names
    text_cols   : list[str] — text column names

    Returns
    -------
    list of two strings: [all_ratings, all_responses]
    """
    all_ratings   = ""
    all_responses = ""

    for cluster_id in range(best_k):
        formatted = generate_formatted_responses(
            labeled_df, cluster_id, likert_cols, text_cols
        )
        all_ratings   += formatted[0] + "\n\n"
        all_responses += formatted[1] + "\n\n"

    return [all_ratings, all_responses]
