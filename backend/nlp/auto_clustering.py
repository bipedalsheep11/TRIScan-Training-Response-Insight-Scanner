# backend/nlp/auto_clustering.py
# ─────────────────────────────────────────────────────────────────
# Clustering pipeline: separates Likert from text columns,
# normalises ratings, embeds text, combines features, reduces
# dimensions with PCA, and runs K-Means with silhouette scoring.
# ─────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer


# ── Column Detection ─────────────────────────────────────────────

def separate_likert_from_text(
    dataframe: pd.DataFrame,
    likert_max_unique: int = 10,
) -> dict:
    """
    Classify each column in the DataFrame as Likert-scale,
    free-text, or other.

    A column is classified as Likert if:
      - Its values are numeric (or can be parsed as numeric)
      - It has few unique values (≤ likert_max_unique)
      - It contains no alphabetic characters

    A column is classified as text if:
      - It contains alphabetic characters
      - Average response length exceeds 15 characters

    Parameters
    ----------
    dataframe        : pd.DataFrame — the full survey dataset
    likert_max_unique: int — upper bound on unique values for
                       a column to be considered Likert-scale

    Returns
    -------
    dict with keys: 'likert', 'text', 'other'
      Each value is a list of column name strings.
    """
    likert_columns = []
    text_columns   = []
    other_columns  = []

    for col_name in dataframe.columns:
        # .dropna() removes missing values so they don't affect detection
        column_data = dataframe[col_name].dropna()

        if len(column_data) == 0:
            other_columns.append(col_name)
            continue

        # Check 1: are values numeric?
        is_numeric = pd.api.types.is_numeric_dtype(column_data)

        # Check 2: how many distinct values? Likert scales are bounded.
        # Note: original code had a bug — unique() returns an array, not a count.
        # We fix that here by using .nunique() which returns the integer count.
        unique_count = column_data.nunique()

        # Check 3: does the column contain any letters?
        has_letters = column_data.astype(str).str.contains("[a-zA-Z]").any()

        # Check 4: what is the average character length?
        avg_len = column_data.astype(str).str.len().mean()

        # ── Decision logic ──
        if is_numeric and not has_letters and unique_count <= likert_max_unique:
            likert_columns.append(col_name)
        elif has_letters and avg_len > 15:
            text_columns.append(col_name)
        else:
            other_columns.append(col_name)

    return {
        "likert": likert_columns,
        "text":   text_columns,
        "other":  other_columns,
    }


# ── Normalisation ────────────────────────────────────────────────

def normalize_likert(likert_df: pd.DataFrame) -> np.ndarray:
    """
    Rescale each Likert column to the range [0.0, 1.0].

    MinMaxScaler maps the column's minimum to 0 and maximum to 1,
    with all values in between scaled proportionally.
    This ensures a 1–5 scale and a 1–7 scale carry equal weight
    when combined with text embeddings.

    Parameters
    ----------
    likert_df : pd.DataFrame — subset of DataFrame with Likert columns only

    Returns
    -------
    np.ndarray, shape (n_respondents, n_likert_columns)
    """
    scaler           = MinMaxScaler()
    normalized_array = scaler.fit_transform(likert_df.fillna(likert_df.mean()))
    print(f"Normalized Likert array shape: {normalized_array.shape}")
    return normalized_array


# ── Text Embeddings ──────────────────────────────────────────────

def embed_text_responses(text_df: pd.DataFrame) -> np.ndarray:
    """
    Convert free-text responses into dense numerical vectors.

    Each respondent's text columns are concatenated into a single
    string, which is then passed through the SentenceTransformer
    model to produce a 384-dimensional embedding vector.

    The model 'all-MiniLM-L6-v2':
      - Is ~80 MB, downloaded automatically on first use
      - Runs on CPU without a GPU
      - Produces embeddings optimised for semantic similarity

    Parameters
    ----------
    text_df : pd.DataFrame — subset of DataFrame with text columns only

    Returns
    -------
    np.ndarray, shape (n_respondents, 384)
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Concatenate all text columns per row into one string.
    # .fillna("") prevents NaN from causing errors during join.
    combined_text = text_df.fillna("").agg(" ".join, axis=1).tolist()

    embeddings = model.encode(combined_text, show_progress_bar=True)
    print(f"Text embedding array shape: {embeddings.shape}")
    return embeddings


# ── Feature Combination ──────────────────────────────────────────

def combine_features(
    normalized_likert: np.ndarray,
    text_embeddings:   np.ndarray,
) -> np.ndarray:
    """
    Horizontally stack Likert and embedding features into one array.

    np.hstack() joins arrays column-wise:
      [n_respondents, n_likert] + [n_respondents, 384]
      → [n_respondents, n_likert + 384]

    Parameters
    ----------
    normalized_likert : np.ndarray — output of normalize_likert()
    text_embeddings   : np.ndarray — output of embed_text_responses()

    Returns
    -------
    np.ndarray, shape (n_respondents, n_likert + 384)
    """
    combined = np.hstack([normalized_likert, text_embeddings])
    print(f"Combined feature array shape: {combined.shape}")
    return combined


# ── Dimensionality Reduction ─────────────────────────────────────

def reduce_dimensions(
    combined_features: np.ndarray,
    n_components:      int = 50,
) -> np.ndarray:
    """
    Reduce the combined feature matrix using PCA.

    PCA (Principal Component Analysis) identifies the directions
    in which the data varies the most and projects all data points
    onto those directions. The result is a lower-dimensional matrix
    that retains most of the meaningful variation.

    Why reduce? K-Means clustering degrades in very high-dimensional
    spaces. Reducing to ~50 dimensions makes clustering more stable.

    Parameters
    ----------
    combined_features : np.ndarray — output of combine_features()
    n_components      : int — number of dimensions to keep (capped
                        at the number of input features)

    Returns
    -------
    np.ndarray, shape (n_respondents, n_components)
    """
    # Guard: you cannot reduce to more dimensions than you started with
    n_components = min(n_components, combined_features.shape[1], combined_features.shape[0] - 1)

    pca     = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(combined_features)

    # explained_variance_ratio_ tells us what fraction of the total
    # variation in the data is captured by the kept components.
    variance_retained = sum(pca.explained_variance_ratio_) * 100
    print(f"Reduced to {n_components} dims, retaining {variance_retained:.1f}% variance.")
    return reduced


# ── K-Means with Silhouette Scoring ─────────────────────────────

def auto_cluster(
    features: np.ndarray,
    min_k:    int = 2,
    max_k:    int = 8,
) -> tuple[int, np.ndarray, dict]:
    """
    Automatically find the best number of clusters by evaluating
    silhouette scores for each candidate k.

    The silhouette score measures how similar each point is to its
    own cluster versus neighbouring clusters. Scores range from -1
    to +1; higher is better, meaning clusters are well-separated.

    Parameters
    ----------
    features : np.ndarray — reduced feature matrix
    min_k    : int — minimum clusters to test (default 2)
    max_k    : int — maximum clusters to test (default 8)

    Returns
    -------
    best_k      : int — number of clusters with highest silhouette score
    best_labels : np.ndarray — cluster assignment per respondent (0-indexed)
    scores_summary : dict — all scores and the winning k
    """
    scores     = {}
    all_labels = {}

    # Cap max_k so we don't try to create more clusters than data points
    max_k = min(max_k, features.shape[0] - 1)

    for k in range(min_k, max_k + 1):
        # n_init=10: run K-Means 10 times with different starting points
        # and keep the result with the lowest inertia (tightest clusters).
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(features)
        score  = silhouette_score(features, labels)

        scores[k]     = score
        all_labels[k] = labels
        print(f"  k={k:2d}  →  silhouette: {score:.4f}")

    best_k      = max(scores, key=scores.get)
    best_labels = all_labels[best_k]

    print(f"\n✓ Best k = {best_k}  (silhouette: {scores[best_k]:.4f})")

    scores_summary = {
        "best_k":    best_k,
        "best_score": scores[best_k],
        "all_scores": scores,
    }
    return best_k, best_labels, scores_summary


# ── Attach Cluster Labels to DataFrame ──────────────────────────

def attach_clusters(
    dataframe:      pd.DataFrame,
    cluster_labels: np.ndarray,
) -> pd.DataFrame:
    """
    Add a 'cluster' column to the DataFrame with integer cluster IDs.

    Parameters
    ----------
    dataframe      : pd.DataFrame — original survey data
    cluster_labels : np.ndarray — output of auto_cluster()

    Returns
    -------
    pd.DataFrame — copy with a new 'cluster' column appended
    """
    labeled_df            = dataframe.copy()
    labeled_df["cluster"] = cluster_labels
    return labeled_df


# ── Full Pipeline Orchestrator ───────────────────────────────────

def run_clustering_pipeline(
    dataframe:     pd.DataFrame,
    force_k:       int | None = None,
    n_pca_dims:    int        = 30,
    min_k:         int        = 2,
    max_k:         int        = 8,
) -> dict:
    """
    Run the full clustering pipeline end-to-end.

    Steps:
      1. Detect and separate Likert vs. text columns
      2. Normalise Likert ratings (MinMaxScaler)
      3. Embed text responses (SentenceTransformer)
      4. Combine features (hstack)
      5. Reduce dimensions (PCA)
      6. Find best k and cluster (K-Means + silhouette)
      7. Attach cluster IDs to the original DataFrame

    Parameters
    ----------
    dataframe  : pd.DataFrame — raw survey data (no preprocessing required)
    force_k    : int | None — if set, skips auto-detection and uses this k
    n_pca_dims : int — number of PCA dimensions to retain (default 30)
    min_k      : int — minimum k to evaluate (default 2)
    max_k      : int — maximum k to evaluate (default 8)

    Returns
    -------
    dict with keys:
      'labeled_df'   : pd.DataFrame — original data with 'cluster' column
      'best_k'       : int — number of clusters used
      'likert_cols'  : list[str] — detected Likert column names
      'text_cols'    : list[str] — detected text column names
      'scores'       : dict — silhouette scores per k (if auto-detected)
      'pca_coords'   : np.ndarray — 2D coords for scatter plot (first 2 PCA dims)
    """
    print("\n── Stage 1: Column Detection ──")
    col_types  = separate_likert_from_text(dataframe)
    likert_cols = col_types["likert"]
    text_cols   = col_types["text"]
    print(f"  Likert: {likert_cols}")
    print(f"  Text:   {text_cols}")

    if not likert_cols and not text_cols:
        raise ValueError(
            "Could not detect any Likert or text columns. "
            "Check that your CSV has numeric rating columns and/or text response columns."
        )

    print("\n── Stage 2: Normalise Likert ──")
    feature_parts = []
    if likert_cols:
        likert_arr = normalize_likert(dataframe[likert_cols])
        feature_parts.append(likert_arr)

    print("\n── Stage 3: Embed Text ──")
    if text_cols:
        text_arr = embed_text_responses(dataframe[text_cols])
        feature_parts.append(text_arr)

    print("\n── Stage 4: Combine Features ──")
    combined = combine_features(*feature_parts) if len(feature_parts) > 1 else feature_parts[0]

    print("\n── Stage 5: PCA Reduction ──")
    reduced = reduce_dimensions(combined, n_pca_dims)
    # Keep first 2 components for the scatter plot visualisation
    pca_2d  = reduce_dimensions(combined, 2)

    print("\n── Stage 6: Clustering ──")
    if force_k is not None:
        kmeans     = KMeans(n_clusters=force_k, n_init=10, random_state=42)
        labels     = kmeans.fit_predict(reduced)
        best_k     = force_k
        scores     = {"best_k": force_k, "best_score": silhouette_score(reduced, labels), "all_scores": {force_k: silhouette_score(reduced, labels)}}
    else:
        best_k, labels, scores = auto_cluster(reduced, min_k=min_k, max_k=max_k)

    print("\n── Stage 7: Attach Labels ──")
    labeled_df = attach_clusters(dataframe, labels)

    return {
        "labeled_df":  labeled_df,
        "best_k":      best_k,
        "likert_cols": likert_cols,
        "text_cols":   text_cols,
        "scores":      scores,
        "pca_coords":  pca_2d,
    }
