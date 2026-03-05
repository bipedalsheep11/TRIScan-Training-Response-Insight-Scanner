# Feedback Analysis System

An AI-powered post-program evaluation analysis tool that processes survey data — both Likert-scale ratings and free-text responses — and produces structured, actionable insights through automatic clustering, sentiment analysis, thematic grouping, and improvement extraction.

---

## Project Structure

```
feedback-analysis-system/
│
├── app.py                          # Streamlit application (Python desktop app)
│
├── frontend/                       # Standalone HTML app (no Python required)
│   ├── index.html                  # Main application shell
│   ├── styles.css                  # Visual design system
│   └── app.js                      # Full application logic
│
├── backend/
│   ├── nlp/
│   │   ├── llm_client.py           # Unified LLM client (Anthropic → Groq → Ollama)
│   │   ├── auto_clustering.py      # Embedding, PCA, K-Means clustering pipeline
│   │   ├── format_responses.py     # Convert DataFrame to LLM-readable strings
│   │   └── analysis_modules.py     # Label clusters, sentiment, themes, insights
│   │
│   └── utils/
│       ├── get_system_prompt.py    # Build the analyst system prompt
│       └── document_reader.py      # Load PDF, DOCX, CSV, XLSX files
│
├── requirements.txt                # Python dependencies
├── .env.example                    # API key template
└── README.md
```

---

## Two Ways to Run

### Option A — HTML App (no Python required)

1. Open `frontend/index.html` in any modern browser.
2. The app connects directly to the **Anthropic API** from the browser.
   - You will need an `ANTHROPIC_API_KEY` set in the app's config or entered manually.
   - Note: The clustering stage (embeddings + K-Means) is simulated in the browser. For production-quality clustering, use Option B.

### Option B — Streamlit App (Python, full pipeline)

This version runs the **full Python pipeline** including real sentence embeddings and K-Means clustering.

#### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Python version:** 3.10 or higher is required.
> The `sentence-transformers` package will download a ~80MB model on first run.

#### 2. Configure API keys

```bash
cp .env.example .env
```

Then edit `.env` and add at least one API key:

| Key | Provider | Notes |
|-----|----------|-------|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | Primary. Recommended. |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | Free tier available. Fast. |
| — | Ollama (local) | No key needed. Run `ollama serve` locally. |

The system tries backends in order: Anthropic → Groq → Ollama. If all fail, the pipeline will raise an error.

#### 3. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Pipeline Stages

| Stage | What happens |
|-------|-------------|
| **1. Data Ingestion** | CSV parsed, Likert and text columns auto-detected |
| **2. Clustering** | Text embedded (all-MiniLM-L6-v2), Likert normalised, combined, PCA-reduced, K-Means with silhouette scoring |
| **3. Cluster Labelling** | AI characterises each cluster: label, respondent profile, key drivers, distinguishing feature |
| **4. Sentiment Analysis** | Per-respondent classification: positive / negative / neutral / mixed, confidence, urgent flags |
| **5. Thematic Clustering** | AI groups responses into recurring themes (auto-discover or predefined) |
| **6. Insight Extraction** | AI surfaces concrete actionable improvement suggestions with priority and breadth |

---

## CSV Format

Your survey CSV should have:

- **Likert columns**: numeric values (e.g. 1–5 scale). Any column with only numbers and ≤10 unique values is auto-detected.
- **Text columns**: free-text responses. Columns with average response length > 20 characters are auto-detected.
- A header row is required.

Example:

```csv
respondent_id,content_rating,facilitator_rating,pacing_rating,what_worked_well,what_could_improve
1,5,5,4,"The facilitator was excellent","Could have more case studies"
2,3,4,2,"Good content","Sessions moved too quickly"
```

---

## Output: Cluster Label JSON

Each cluster produces a JSON object from `label_cluster_with_llm()`:

```json
{
  "label": "Engaged Advocates",
  "respondent_profile": "Highly engaged participants who found strong alignment...",
  "key_drivers": ["Facilitator delivery", "Content depth", "Practical applicability"],
  "distinguishing_features": "This cluster shows the highest ratings and expresses genuine advocacy."
}
```

## Output: Sentiment JSON

Each respondent produces a record from `analyze_sentiment()`:

```json
{
  "cluster": 0,
  "respondent_id": "R001",
  "sentiment": "positive",
  "confidence": "high",
  "flag_urgent": false,
  "flag_reason": null,
  "key_phrases": ["facilitator was excellent", "content directly applicable", "best training attended"]
}
```

---

## GitHub Setup

```bash
git init
git add .
git commit -m "Initial commit: Feedback Analysis System"
git remote add origin https://github.com/YOUR_USERNAME/feedback-analysis-system.git
git push -u origin main
```

**Important:** Never commit your `.env` file. It is already in `.gitignore` below.

### Recommended `.gitignore`

```
.env
__pycache__/
*.pyc
.DS_Store
*.egg-info/
dist/
build/
.streamlit/secrets.toml
```

---

## Dependencies Explained

| Package | Purpose |
|---------|---------|
| `streamlit` | Python web UI framework — turns scripts into interactive apps |
| `pandas` | DataFrame manipulation for CSV loading and column operations |
| `numpy` | Numerical arrays used by scikit-learn and the clustering pipeline |
| `scikit-learn` | MinMaxScaler, PCA, KMeans, silhouette_score |
| `sentence-transformers` | Loads `all-MiniLM-L6-v2` to embed text as vectors |
| `plotly` | Interactive charts in the Streamlit dashboard |
| `anthropic` | Official Anthropic Python SDK for Claude API calls |
| `groq` | Groq Python SDK for fast LLM inference (fallback) |
| `ollama` | Local Ollama Python client (offline fallback) |
| `pymupdf` | PDF text extraction (`fitz`) |
| `python-docx` | DOCX text extraction |
| `python-dotenv` | Loads `.env` file into environment variables |

---

## Licence

MIT — free to use, modify, and distribute.
