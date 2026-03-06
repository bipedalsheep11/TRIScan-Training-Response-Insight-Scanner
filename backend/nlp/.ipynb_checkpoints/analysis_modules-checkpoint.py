# backend/nlp/analysis_modules.py
# ─────────────────────────────────────────────────────────────────
# Three AI-powered analysis stages:
#   1. label_clusters     — characterise each cluster with the LLM
#   2. analyze_sentiment  — per-respondent sentiment + urgent flags
#   3. cluster_themes     — thematic grouping of responses
#   4. extract_insights   — actionable improvement suggestions
# ─────────────────────────────────────────────────────────────────

import json
import pandas as pd
from backend.nlp.format_responses   import get_all_clusters_table, generate_formatted_responses
from .llm_client import call_llm_with_retry, parse_llm_json


# ════════════════════════════════════════════════════════════════
# 1. CLUSTER LABELLING
# ════════════════════════════════════════════════════════════════

def label_cluster_with_llm(
    cluster_id:         int,
    formatted_ratings:  str,
    formatted_responses:str,
    all_clusters_table: list[str],
    system_prompt:      str,
) -> dict:
    """
    Ask the LLM to characterise a single cluster.

    We provide two sources of context:
      - all_clusters_table: ratings and responses for ALL clusters,
        so the model can make comparisons and ensure each label
        is meaningfully distinct from the others.
      - formatted_ratings + formatted_responses: the specific data
        for the cluster being labelled.

    Parameters
    ----------
    cluster_id          : int — index of the cluster to label
    formatted_ratings   : str — average ratings for this cluster
    formatted_responses : str — sample text responses for this cluster
    all_clusters_table  : list[str] — [all_ratings, all_responses]
    system_prompt       : str — analyst role + output constraints

    Returns
    -------
    dict with keys:
      label, respondent_profile, key_drivers, distinguishing_features
    """
    user_prompt = f"""You are analyzing post-program training evaluation data.
Your task is to characterize the respondents in Cluster {cluster_id} —
describe who they are as a group, based on their ratings and written responses.

Characterization must focus on:
- Their overall satisfaction and engagement with the training
- Which aspects they responded most strongly to (facilitator quality,
  pacing, content relevance, practical applicability)
- Consistent patterns in what they valued or found lacking

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT — All clusters (for comparison):

All clusters — average ratings:
{all_clusters_table[0]}

All clusters — sample text responses:
{all_clusters_table[1]}

Treat the responses as answers to the column header questions.
Always reference what question the respondent was answering.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOU ARE LABELLING: Cluster {cluster_id}

Cluster {cluster_id} — average ratings:
{formatted_ratings}

Cluster {cluster_id} — sample text responses:
{formatted_responses}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a JSON object. No markdown. No code fences. No commentary.

{{
  "label": "A short 2-4 word phrase that distinctly characterizes this group",
  "respondent_profile": "One to two sentences describing who these respondents appear to be — their overall disposition toward the training, and what most shaped their experience",
  "key_drivers": ["The 2-4 specific training aspects that most explain this cluster's ratings and responses, drawn directly from the data"],
  "distinguishing_features": "One sentence explaining what makes this cluster different from the other clusters shown above"
}}"""

    raw    = call_llm_with_retry(system_prompt, user_prompt, max_tokens=600)
    result = parse_llm_json(raw)

    if not result:
        # Return a minimal valid dict so the pipeline does not crash
        return {
            "label": f"Cluster {cluster_id}",
            "respondent_profile": "Could not generate profile — LLM response could not be parsed.",
            "key_drivers": [],
            "distinguishing_features": "Parse error — review raw LLM output.",
            "_raw": raw,
        }
    return result


def label_all_clusters(
    best_k:             int,
    labeled_df:         pd.DataFrame,
    likert_cols:        list[str],
    text_cols:          list[str],
    all_clusters_table: list[str],
    system_prompt:      str,
) -> dict:
    """
    Label all clusters by calling label_cluster_with_llm() for each one.

    Parameters
    ----------
    best_k              : int — number of clusters
    labeled_df          : pd.DataFrame — data with 'cluster' column
    likert_cols         : list[str] — Likert column names
    text_cols           : list[str] — text column names
    all_clusters_table  : list[str] — context for comparison
    system_prompt       : str — LLM system prompt

    Returns
    -------
    dict keyed by cluster index (str): "0", "1", …
    """
    from .format_responses import generate_formatted_responses

    labels = {}
    for cluster_id in range(best_k):
        print(f"  Labelling Cluster {cluster_id}…")
        formatted = generate_formatted_responses(
            labeled_df, cluster_id, likert_cols, text_cols
        )
        labels[str(cluster_id)] = label_cluster_with_llm(
            cluster_id          = cluster_id,
            formatted_ratings   = formatted[0],
            formatted_responses = formatted[1],
            all_clusters_table  = all_clusters_table,
            system_prompt       = system_prompt,
        )
    return labels


# ════════════════════════════════════════════════════════════════
# 2. SENTIMENT ANALYSIS
# ════════════════════════════════════════════════════════════════

def analyze_sentiment(
    labeled_df: pd.DataFrame,
    best_k: int, 
    likert_cols: list, text_cols: list,
    system_prompt:          str,
) -> dict:
    """
    Classify sentiment for every respondent across all clusters.

    Parameters
    ----------
    all_clusters_responses : list[str] — [avg_ratings_block, responses_block]
    system_prompt          : str — LLM system prompt

    Returns
    -------
    dict with keys:
      total_classified, results (list of per-respondent dicts),
      cluster_summary (sentiment % per cluster)
    """
    all_results = []
    # repeat for all clusters
    for cluster_id in range(best_k):
        # Split the text into a list of lines, ignoring any trailing empty lines
        lines = generate_formatted_responses(labeled_df, cluster_id, likert_cols, text_cols)[1].strip().split('\n')
        
        # If there's no data (need at least title, divider, header, and 1 row = 4 lines)
        if not lines or len(lines) < 4:
            continue # Skip to the next cluster instead of returning out of the whole function
            
        # The actual CSV header is at index 2
        header = lines[2]
        
        # List to hold our new segmented strings
        segments = []
        
        # The actual data rows start at index 3.
        # GUARD: a valid data row must start with a respondent number — either
        # a bare integer ("1", "42") or the R-prefixed zero-padded form ("R001").
        # Any line that does NOT match this pattern is a phantom produced by an
        # embedded newline inside a cell value that slipped through sanitization
        # (e.g. pasted text with line breaks, Windows \r\n from an older export).
        # Silently skipping it prevents that fragment from being sent to the LLM
        # as a separate respondent and inflating n_classified.
        import re
        _valid_row = re.compile(r'^(R\d+|\d+)\s*,')

        for line in lines[3:]:
            if not _valid_row.match(line.strip()):
                # This line is a continuation fragment, not a new respondent row.
                # Log it so the developer can see it, but do not create a segment.
                print(f"  [sentiment] Skipping phantom line in cluster {cluster_id}: {repr(line[:80])}")
                continue
            segments.append(f"{header}\n{line}")
            
        # Loop through each segmented line and call the LLM
        for i, segment in enumerate(segments, start=1):
            user_prompt = f"""Analyze the sentiment of the respondent shown below. 
            
            Respondent responses in CSV format (columns: Respondent Number, then question columns). Treat this as a Comma-Separated Value (CSV) file:
            {segment}
            
            For this respondent, classify their sentiment and return a JSON object with this exact structure.
            Return ONLY the JSON. No markdown. No code fences.
            
            {{
              "results": [
                {{
                  "cluster": <cluster number as integer, use 0 if unknown>,
                  "respondent_id": "<respondent number>",
                  "sentiment": "<positive | negative | neutral | mixed>",
                  "confidence": "<high | medium | low>",
                  "flag_urgent": <true | false>,
                  "flag_reason": "<one sentence if urgent, otherwise null>",
                  "key_phrases": ["<3-6 word phrase>", "<3-6 word phrase>", "<3-6 word phrase>"]
                }}
              ]
            }}
            
            SENTIMENT DEFINITIONS:
            - positive: satisfaction, appreciation, or benefit from training
            - negative: dissatisfaction, frustration, or significant problems
            - neutral:  factual observations without clear valence
            - mixed:    both positive and negative in the same response
            
            FLAG URGENT if ANY of these apply:
            - Strong dissatisfaction that would damage programme credibility if repeated
            - Logistical failure that prevented meaningful participation
            - Safety, health, or welfare concern
            - Explicit refusal to recommend or return to this programme
            
            KEY PHRASES: extract up to 3 short phrases (3-6 words each) capturing the core of the response."""
            
            try:
                # Note: Ensure you have your JSON parsing and LLM calling functions imported
                raw = call_llm_with_retry(system_prompt, user_prompt, max_tokens=1000)
                parsed_raw = parse_llm_json(raw)
                
                # Aggregate the result.
                # Override the LLM's cluster guess with the authoritative value
                # from the loop — the LLM receives no cluster column so it always
                # defaults to 0, which breaks cluster_summary for every cluster > 0.
                if parsed_raw and "results" in parsed_raw:
                    for r in parsed_raw["results"]:
                        r["cluster"] = cluster_id
                    all_results.extend(parsed_raw["results"])
                    
            except Exception as e:
                print(f"Failed to classify row {i}: {e}")
                continue

    # ── Initialize final result structure
    result = {
        "total_classified": len(all_results),
        "results": all_results,
        "cluster_summary": {}
    }

    if not all_results:
        return result

    # ── Recompute cluster_summary from the compiled results list
    cluster_counts: dict[str, dict[str, int]] = {}
    for r in all_results:
        ckey = str(r.get("cluster", "0"))
        if ckey not in cluster_counts:
            cluster_counts[ckey] = {"positive": 0, "neutral": 0, "negative": 0, "mixed": 0}
            
        sentiment = r.get("sentiment", "neutral").lower()
        if sentiment in cluster_counts[ckey]:
            cluster_counts[ckey][sentiment] += 1
        else:
            cluster_counts[ckey]["neutral"] += 1 # fallback

    # Convert raw counts to rounded percentages per cluster
    cluster_summary: dict[str, dict[str, int]] = {}
    for ckey, counts in cluster_counts.items():
        total_in_cluster = sum(counts.values())
        if total_in_cluster == 0:
            cluster_summary[ckey] = {"positive": 0, "neutral": 0, "negative": 0, "mixed": 0}
        else:
            cluster_summary[ckey] = {
                sentiment: round((count / total_in_cluster) * 100)
                for sentiment, count in counts.items()
            }

    result["cluster_summary"] = cluster_summary
    return result




# ════════════════════════════════════════════════════════════════
# 3. THEMATIC CLUSTERING
# ════════════════════════════════════════════════════════════════

def cluster_themes(
    all_clusters_responses: list[str],
    system_prompt:          str,
    predefined_themes:      list[str] | None = None,
) -> dict:
    """
    Group responses into recurring themes.

    Two modes:
      - Predefined: caller supplies theme labels; model assigns each
        response to the closest label (or "Other").
      - Discovery:  model identifies 3-8 themes from the data itself.

    Parameters
    ----------
    all_clusters_responses : list[str] — [ratings_block, responses_block]
    system_prompt          : str
    predefined_themes      : list[str] | None — if provided, uses these

    Returns
    -------
    dict with keys:
      themes (list of theme dicts), coded_responses (list of assignments)
    """
    if predefined_themes:
        theme_list = "\n".join(f"- {t}" for t in predefined_themes)
        theme_instruction = f"""Assign each response to ONE of these predefined themes.
        If a response fits none, assign it to "Other" and note the actual theme it represents.
        
        PREDEFINED THEMES:
        {theme_list}"""
    else:
        theme_instruction = """Identify 4-8 recurring themes from the responses.
        Name each theme as a short noun phrase (e.g. "Facilitator Clarity", "Module Pacing").
        Do not create a unique theme per response — look for patterns across respondents."""
        
        user_prompt = f"""You are performing thematic analysis on post-training evaluation responses.
        
        {theme_instruction}
        
        RESPONDENT RESPONSES:
        {all_clusters_responses[1]}
        
        Return ONLY a JSON object. No markdown. No code fences.
        
        {{
          "themes": [
            {{
              "name": "Theme Name",
              "count": <integer — number of responses assigned to this theme>,
              "description": "One sentence characterising what respondents in this theme are saying",
              "clusters": [<list of cluster IDs where this theme appears>]
            }}
          ]
        }}"""

    raw    = call_llm_with_retry(system_prompt, user_prompt, max_tokens=2500)
    result = parse_llm_json(raw)

    if not result:
        return {
            "themes": [],
            "_parse_error": raw,
        }
    return result


# ════════════════════════════════════════════════════════════════
# 4. ACTIONABLE INSIGHT EXTRACTION
# ════════════════════════════════════════════════════════════════

def extract_actionable_insights(
    all_clusters_responses: list[str],
    system_prompt:          str,
) -> dict:
    """
    Extract concrete, actionable improvement suggestions from all
    clusters simultaneously.

    Analysing all clusters together allows the model to identify
    whether a suggestion is isolated (one cluster only) or systemic
    (multiple clusters), which determines priority.

    Parameters
    ----------
    all_clusters_responses : list[str] — [ratings_block, responses_block]
    system_prompt          : str

    Returns
    -------
    dict with keys:
      total_insights, insights (list), priority_summary (dict)
    """
    user_prompt = f"""Extract concrete, actionable improvement suggestions from these
post-training evaluation responses.

An actionable insight must be:
- Grounded in something a participant explicitly said or clearly implied
- Specific enough to guide a real decision (not vague praise or complaint)
- Distinct from other insights — do not repeat the same suggestion

Do NOT include:
- General positive feedback with no improvement implication
- Vague statements like "make it better"
- Observations that describe a problem without any implied solution

EVALUATION RESPONSES BY CLUSTER:
{all_clusters_responses[1]}

Return ONLY a JSON object. No markdown. No code fences.

{{
  "total_insights": <integer>,
  "insights": [
    {{
      "id": "INS-001",
      "priority": "<high | medium | low>",
      "category": "<Facilitator | Content | Pacing | Logistics | Materials | Assessment | Follow-up | Other>",
      "insight": "<one clear actionable recommendation sentence — written as a recommendation, e.g. 'Provide printed handouts…'>",
      "source_clusters": [<cluster_id>, ...],
      "evidence": "<paraphrased summary of participant comments supporting this insight>",
      "breadth": "<isolated | recurring | widespread>"
    }}
  ],
  "priority_summary": {{
    "high":   "<one sentence summarising the high-priority theme>",
    "medium": "<one sentence summarising the medium-priority theme>",
    "low":    "<one sentence summarising the low-priority theme>"
  }}
}}

PRIORITY:
- high:   takes relatively low effort to implement compared to the impact when implement. The impact is widespread and is very significant in the context of the program.
- medium: takes relatively equal effort to the impact when implemented. The impact is valuable but not as widespread and significant in the context of the program
- low:    takes relatively high effort compared to the impact when implemented. The impact is isolated and not very significant in the context of the program.

BREADTH:
- isolated:   one cluster only
- recurring:  two clusters
- widespread: three or more clusters

Sort: priority descending (high first), then breadth (widespread before isolated)."""

    raw    = call_llm_with_retry(system_prompt, user_prompt, max_tokens=3000)
    result = parse_llm_json(raw)

    if not result:
        return {
            "total_insights": 0,
            "insights": [],
            "priority_summary": {"high": "", "medium": "", "low": ""},
            "_parse_error": raw,
        }
    return result
