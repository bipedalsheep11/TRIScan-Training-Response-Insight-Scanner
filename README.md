# ◉ Feedback Analysis System

> AI-powered post-program evaluation analytics — cluster respondents, surface sentiment, discover themes, and extract actionable insights from survey data.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [CSV Format Guide](#csv-format-guide)
5. [Configuration](#configuration)
6. [The Six-Stage Pipeline](#the-six-stage-pipeline)
7. [Application Pages](#application-pages)
8. [LLM Backends](#llm-backends)
9. [Dependencies](#dependencies)
10. [Known Issues](#known-issues)
11. [Known Limitations](#known-limitations)
12. [Contributing](#contributing)

---

## What It Does

The Feedback Analysis System takes a post-program evaluation survey — the kind distributed after a training session, workshop, or course — and runs it through a six-stage AI pipeline that produces:

- **Respondent clusters** — groups of participants with similar patterns of ratings and written responses, labeled and profiled by an LLM
- **Per-respondent sentiment** — positive / negative / neutral / mixed, with urgent-flag logic for responses that need immediate attention
- **Recurring themes** — either discovered automatically from the data, or matched against categories you define
- **Actionable insights** — specific, ranked improvement suggestions extracted from participant comments, sorted by priority and how widely they appear across clusters
- **Interactive dashboard** — KPI metrics, 2D cluster scatter plot, sentiment chart, theme frequency bars, and a filterable respondent table
- **Ask AI** — a chat interface loaded with full analysis context so you can query your results in natural language

---

## Quick Start

### Option A — One-click launcher (recommended for non-developers)

Double-click **`run.py`**, or from a terminal:

```bash
python run.py
```

This will automatically check your Python version, create a virtual environment, install all dependencies, prompt for your API key if `.env` is missing, and open the app in your browser.

### Option B — Manual setup

```bash
# 1. Download and unzip the project, then open a terminal in its folder

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# Open .env in any text editor and add your key

# 5. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**.

> **First-run note:** The sentence-transformers model (`all-MiniLM-L6-v2`, ~80 MB) downloads automatically on the first pipeline run and is cached locally. Subsequent runs do not re-download it.

---

## Project Structure

```
feedback-analysis-system/
│
├── app.py                          ← Streamlit frontend (entry point)
├── run.py                          ← One-click launcher for non-developers
├── requirements.txt                ← All Python dependencies
├── .env.example                    ← API key template — copy to .env
├── .gitignore
├── README.md
├── Documentation.pdf               ← Full technical reference
│
└── backend/
    ├── __init__.py
    ├── nlp/
    │   ├── __init__.py
    │   ├── llm_client.py           ← Three-tier LLM fallback chain
    │   ├── auto_clustering.py      ← Embedding + PCA + K-Means pipeline
    │   ├── format_responses.py     ← DataFrame → LLM-readable text formatter
    │   └── analysis_modules.py     ← Labelling / Sentiment / Themes / Insights
    │
    └── utils/
        ├── __init__.py
        ├── get_system_prompt.py    ← System prompt builder
        └── document_reader.py      ← PDF / DOCX / TXT / CSV / XLSX loader
```

The `backend/` package uses relative imports (e.g. `from .llm_client import ...`). The `__init__.py` files in each subdirectory are required for Python to treat them as packages — do not delete them.

---

## CSV Format Guide

Your survey CSV should have at least one column of each type:

| Column type | How it is detected | Typical column names |
|---|---|---|
| **Likert** | Numeric values, ≤ 10 unique values, no letters | `Q1_Facilitation`, `Overall_Rating` |
| **Text** | Contains letters, average response length > 15 chars | `What_Worked`, `Suggestions`, `Comments` |

**Auto-dropped columns:** A leading index column named `""`, `Unnamed: 0`, or `index` is automatically removed before processing.

**Minimum rows:** At least 4 respondents are needed for K-Means to evaluate k = 2. Use manual k mode with k = 2 for very small surveys.

**Example structure:**

```
Q1_Facilitation,Q2_Content,Q3_Pacing,Q4_Overall,What worked well,What could improve
5,4,3,4,The facilitator was very engaging,The afternoon sessions felt rushed
4,5,4,5,Practical examples were excellent,More time for group discussion
3,3,2,3,Good content overall,Pacing was too fast for complex topics
```

**Comma handling:** Internal commas within a text response are replaced with semicolons when building LLM context. This is intentional — it prevents column-alignment errors in the pseudo-CSV format sent to the model.

---

## Configuration

Copy `.env.example` to `.env` and add at least one key:

```bash
cp .env.example .env
```

```ini
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
```

You only need one key. The system tries all three backends in order and uses whichever responds first.

### Obtaining API keys

| Backend | Where to sign up | Cost |
|---|---|---|
| Anthropic | https://console.anthropic.com | Pay-as-you-go |
| Groq | https://console.groq.com | Free tier available |
| Ollama | https://ollama.com | Free — runs on your own machine |

For Ollama: install the app, then run `ollama pull qwen3:8b` before starting.

### Changing the model

Edit the constants at the top of `backend/nlp/llm_client.py`:

```python
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
GROQ_MODEL      = "llama-3.3-70b-versatile"
OLLAMA_MODEL    = "qwen3:8b"
```

---

## The Six-Stage Pipeline

```
Stage 1-2  Detect column types → Normalize Likert → Embed text
           → Combine features → PCA reduction → K-Means clustering

Stage 3    LLM labels each cluster
           (name, respondent profile, key drivers, distinguishing feature)

Stage 4    Per-respondent sentiment classification + urgent flags

Stage 5    Theme discovery (auto) or assignment (predefined labels)

Stage 6    Actionable insight extraction + priority ranking
```

Stages 3–6 are individually wrapped in try/except. If one LLM stage fails, the app shows a warning and continues — clustering results are never lost because of a downstream failure.

### Clustering detail (Stages 1–2)

Likert columns are normalized to [0, 1] with Min-Max scaling (missing values filled by column median). Text columns are converted to 384-dimensional vectors by `sentence-transformers/all-MiniLM-L6-v2`, a CPU-friendly semantic model. Both feature sets are horizontally stacked and reduced with PCA (default: 30 components). K-Means tests k = 2 through 8 and keeps the k with the highest silhouette score. A second 2-component PCA pass produces scatter plot coordinates.

---

## Application Pages

| Page | Purpose |
|---|---|
| **Upload & Config** | Upload CSV + optional context document; configure options; run pipeline |
| **Run Pipeline** | Summary: cluster count, respondent count, theme count |
| **Cluster Profiles** | Per-cluster expandable cards with label, profile, drivers, ratings bars, sentiment |
| **Dashboard** | KPI row + four interactive Plotly charts + top action items + urgent flags |
| **Respondent Table** | Filterable table; sentiment, confidence, flag, key phrase columns; detail view |
| **Ask AI** | Natural-language Q&A with full analysis context loaded into the LLM |

---

## LLM Backends

```
Priority 1  Anthropic Claude (claude-sonnet-4-20250514)
            Requires: ANTHROPIC_API_KEY in .env
            Best structured-JSON output reliability

Priority 2  Groq (llama-3.3-70b-versatile)
            Requires: GROQ_API_KEY in .env
            Very fast; suitable for development and testing

Priority 3  Ollama (qwen3:8b)
            Requires: Ollama app running on localhost:11434
            Fully offline; no API cost; slower on consumer hardware
```

The active backend is shown in the sidebar footer. All three backends receive identical prompts.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.32 | Web UI framework |
| `pandas` | ≥ 2.0 | Data loading and manipulation |
| `numpy` | ≥ 1.24 | Array operations |
| `scikit-learn` | ≥ 1.3 | KMeans, PCA, MinMaxScaler, silhouette_score |
| `sentence-transformers` | ≥ 2.6 | Text embeddings (~80 MB model on first run) |
| `plotly` | ≥ 5.18 | Interactive charts |
| `anthropic` | ≥ 0.25 | Anthropic API client |
| `groq` | ≥ 0.9 | Groq API client |
| `ollama` | ≥ 0.2 | Local Ollama client |
| `python-dotenv` | ≥ 1.0 | `.env` file loading |
| `pymupdf` | ≥ 1.23 | PDF text extraction (imported as `fitz`) |
| `python-docx` | ≥ 1.1 | Word `.docx` extraction |
| `openpyxl` | ≥ 3.1 | Required by pandas for `.xlsx` read/write |

---

## Known Issues

These are confirmed bugs in the current codebase. Source files are not modified here — they are documented for transparency and future resolution.

### Bug 1 — Theme analysis crashes when predefined themes are supplied (Critical)

**File:** `backend/nlp/analysis_modules.py` — `cluster_themes()`  
**Effect:** Stage 5 silently produces an empty theme list whenever you enter themes in the predefined themes box.  
**Cause:** The variable `user_prompt` is only defined inside the `else` block (auto-discovery path). When `predefined_themes` is truthy, `user_prompt` is never assigned, so `call_llm_with_retry(system_prompt, user_prompt, ...)` raises `NameError`. The `try/except` in `app.py` catches it silently.  
**Workaround:** Leave the predefined themes box blank — auto-discovery works correctly.

### Bug 2 — Ask AI prompt contains literal `{program_section}` and `{doc_section}` (Moderate)

**File:** `app.py` — `sys_prompt` in the Ask AI page  
**Effect:** The security/scope rules in the Ask AI system prompt contain the raw text `{program_section}` and `{doc_section}` as literal strings rather than the program name and document content.  
**Cause:** The `"""..."""` block appended to `sys_prompt` is a regular string, not an f-string. Python only interpolates `{variable}` in f-strings.

---

## Known Limitations

- **Large surveys (200+ respondents):** Sentiment analysis makes one LLM API call per respondent. 200 respondents = 200 sequential calls. Expect 5–15 minutes on a free-tier API.
- **Multilingual surveys:** The embedding model is optimized for English. For multilingual data, swap to `paraphrase-multilingual-MiniLM-L12-v2` in `auto_clustering.py`.
- **Very small surveys (< 6 respondents):** Use manual k = 2 to avoid silhouette scoring failures.
- **Non-Likert numerics:** Numeric columns with few unique values (e.g. binary flags, year fields) may be misdetected as Likert. Review the detected column types after running.
---

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Add `pytest` tests for new logic in `backend/`
3. Run `python -m pytest` before submitting
4. Open a pull request with a description of what changed and why

Do not commit `.env` files or survey data (CSV/XLSX) to the repository.

---

*Built with Streamlit · scikit-learn · sentence-transformers · Anthropic Claude*
