# backend/nlp/llm_client.py
# ─────────────────────────────────────────────────────────────────
# Unified LLM client with three-tier fallback:
#   1. Anthropic Claude (primary — used by the Streamlit app)
#   2. Groq (fallback — fast cloud inference, free tier available)
#   3. Local Ollama (offline fallback — no internet required)
#
# Which backend is active is tracked in `current_backend` so the
# Streamlit UI can display it in the status bar.
# ─────────────────────────────────────────────────────────────────

import os
import time
import json
from dotenv import load_dotenv

load_dotenv()

# ── Model configuration ───────────────────────────────────────────
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
GROQ_MODEL      = "llama-3.3-70b-versatile"
OLLAMA_MODEL    = "qwen3:8b"

# Tracks which backend was last used — readable by the UI
current_backend: str = "none"


def call_llm(
    system_prompt: str,
    user_prompt:   str,
    max_tokens:    int   = 1500,
    temperature:   float = 0.05,
) -> str:
    """
    Send a prompt to an LLM and return its text response.

    Tries three backends in priority order:
      1. Anthropic Claude  — requires ANTHROPIC_API_KEY in .env
      2. Groq              — requires GROQ_API_KEY in .env
      3. Local Ollama      — requires Ollama running on localhost

    Parameters
    ----------
    system_prompt : str
        The system-level instructions that define the model's role
        and output format constraints.
    user_prompt : str
        The specific task or data the model should process.
    max_tokens : int
        Maximum number of tokens the model may generate.
        Default 1500 is sufficient for most classification tasks.
    temperature : float
        Controls output randomness. 0.05 = near-deterministic,
        which is appropriate for structured JSON classification.

    Returns
    -------
    str
        Raw text response from the model. Callers are responsible
        for parsing JSON if a structured response is expected.

    Raises
    ------
    RuntimeError
        If all three backends fail.
    """
    global current_backend

    # ── Attempt 1: Anthropic Claude ───────────────────────────────
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
            client   = anthropic.Anthropic(api_key=anthropic_key)
            response = client.messages.create(
                model      = ANTHROPIC_MODEL,
                max_tokens = max_tokens,
                system     = system_prompt,
                messages   = [{"role": "user", "content": user_prompt}]
            )
            current_backend = "anthropic"
            # response.content is a list of content blocks; we join all text blocks
            return "".join(block.text for block in response.content if hasattr(block, "text"))
        except Exception as e:
            print(f"⚠ Anthropic unavailable ({type(e).__name__}: {e}). Trying Groq…")

    # ── Attempt 2: Groq ───────────────────────────────────────────
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq
            client   = Groq(api_key=groq_key)
            response = client.chat.completions.create(
                model      = GROQ_MODEL,
                messages   = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens  = max_tokens,
                temperature = temperature,
                timeout     = 30,
            )
            current_backend = "groq"
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠ Groq unavailable ({type(e).__name__}: {e}). Trying Ollama…")

    # ── Attempt 3: Local Ollama ───────────────────────────────────
    try:
        import ollama
        response = ollama.chat(
            model   = OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            options = {"temperature": temperature, "num_predict": max_tokens},
        )
        current_backend = "ollama"
        return response["message"]["content"]
    except Exception as e:
        raise RuntimeError(
            "All three LLM backends failed.\n"
            "  • Anthropic: set ANTHROPIC_API_KEY in .env\n"
            "  • Groq:      set GROQ_API_KEY in .env\n"
            "  • Ollama:    run `ollama serve` locally\n"
            f"Ollama error: {e}"
        )


def get_active_backend() -> str:
    """Return the name of the backend that responded last."""
    return current_backend


def call_llm_with_retry(
    system_prompt: str,
    user_prompt:   str,
    max_tokens:    int   = 10000,
    temperature:   float = 0.02,
    retries:       int   = 3,
) -> str:
    """
    Wrapper around call_llm() that retries transient failures
    with exponential back-off (1s, 2s, 4s between attempts).

    RuntimeError (all backends down) is re-raised immediately
    without retrying — there is nothing to retry in that case.
    """
    for attempt in range(retries):
        try:
            return call_llm(system_prompt, user_prompt, max_tokens, temperature)
        except RuntimeError:
            raise   # All backends failed — no point retrying
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  Attempt {attempt + 1} failed ({e}). Retrying in {wait}s…")
                time.sleep(wait)
            else:
                raise


def parse_llm_json(raw_text: str) -> dict | list | None:
    """
    Safely parse JSON from LLM output.

    LLMs sometimes wrap JSON in markdown code fences (```json … ```).
    This function strips those before parsing so callers do not need
    to handle that themselves.

    Returns None if parsing fails, so callers can fall back gracefully
    without crashing the pipeline.
    """
    if not raw_text:
        return None
    # Strip markdown code fences if present
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines   = cleaned.split("\n")
        cleaned = "\n".join(lines[1:])          # Remove opening ```json line
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]              # Remove closing ```
    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        return None
