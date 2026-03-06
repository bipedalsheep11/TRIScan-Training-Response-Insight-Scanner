#!/usr/bin/env python3
"""
run.py — One-click launcher for the Feedback Analysis System
─────────────────────────────────────────────────────────────
Double-click this file, or run:  python run.py

What this script does, in order:
  1. Checks that Python 3.10 or newer is installed
  2. Creates a virtual environment in ./venv (skips if it already exists)
  3. Installs all dependencies from requirements.txt into the venv
  4. Checks for a .env file; prompts you to enter an API key if one is missing
  5. Launches the Streamlit app inside the venv
  6. Opens http://localhost:8501 in your default browser

No coding knowledge is needed to use this file.
"""

import sys
import os
import subprocess
import platform
import time
import shutil

# ── Terminal colour helpers ───────────────────────────────────────
# These only add colour on macOS/Linux; Windows shows plain text.
def _c(code, text):
    if platform.system() == "Windows":
        return text
    return f"\033[{code}m{text}\033[0m"

def ok(msg):    print(_c("32", f"  ✓  {msg}"))
def info(msg):  print(_c("34", f"  →  {msg}"))
def warn(msg):  print(_c("33", f"  ⚠  {msg}"))
def err(msg):   print(_c("31", f"  ✗  {msg}"))
def head(msg):  print(_c("1",  f"\n{msg}"))
def sep():      print("─" * 56)


# ════════════════════════════════════════════════════════════════
# STEP 1 — Python version check
# ════════════════════════════════════════════════════════════════
def check_python():
    head("Step 1 · Checking Python version")
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}.{sys.version_info[2]}"
    if major < 3 or (major == 3 and minor < 10):
        err(f"Python {version_str} detected. Python 3.10 or newer is required.")
        err("Download the latest version from https://python.org/downloads")
        input("\nPress Enter to exit.")
        sys.exit(1)
    ok(f"Python {version_str}")


# ════════════════════════════════════════════════════════════════
# STEP 2 — Virtual environment
# ════════════════════════════════════════════════════════════════
def setup_venv():
    """
    Create a virtual environment in ./venv if one does not exist.

    A virtual environment is an isolated copy of Python that keeps
    this project's dependencies separate from everything else on
    your computer. This prevents version conflicts.
    """
    head("Step 2 · Virtual environment")

    here   = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(here, "venv")

    if os.path.isdir(venv_dir):
        ok("Virtual environment already exists — skipping creation")
    else:
        info("Creating virtual environment in ./venv …")
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        ok("Virtual environment created")

    # Return paths to the Python and pip executables inside the venv.
    # On Windows they live in venv/Scripts/; on Unix in venv/bin/.
    if platform.system() == "Windows":
        python_path = os.path.join(venv_dir, "Scripts", "python.exe")
        pip_path    = os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        python_path = os.path.join(venv_dir, "bin", "python")
        pip_path    = os.path.join(venv_dir, "bin", "pip")

    return python_path, pip_path


# ════════════════════════════════════════════════════════════════
# STEP 3 — Install dependencies
# ════════════════════════════════════════════════════════════════
def install_requirements(pip_path):
    """
    Install packages listed in requirements.txt into the venv.

    pip is the standard Python package installer. The -q flag
    suppresses verbose output. --upgrade ensures pip itself is
    current before installing project dependencies.
    """
    head("Step 3 · Installing dependencies")

    here = os.path.dirname(os.path.abspath(__file__))
    req  = os.path.join(here, "requirements.txt")

    if not os.path.isfile(req):
        err("requirements.txt not found. Cannot install dependencies.")
        err("Make sure you are running run.py from the project folder.")
        input("\nPress Enter to exit.")
        sys.exit(1)

    # Upgrade pip first — avoids warnings about outdated pip
    info("Upgrading pip …")
    subprocess.run([pip_path, "install", "--upgrade", "pip", "-q"], check=True)

    info("Installing project dependencies (this may take a few minutes on first run) …")
    info("Note: sentence-transformers downloads ~80 MB on the very first run.")
    result = subprocess.run(
        [pip_path, "install", "-r", req, "-q"],
        capture_output=False,
    )
    if result.returncode != 0:
        err("Dependency installation failed. Check the output above for details.")
        input("\nPress Enter to exit.")
        sys.exit(1)
    ok("All dependencies installed")


# ════════════════════════════════════════════════════════════════
# STEP 4 — .env / API key check
# ════════════════════════════════════════════════════════════════
def check_env():
    """
    Verify that a .env file exists with at least one API key.

    The app needs at least one LLM backend to function:
      - Anthropic Claude (ANTHROPIC_API_KEY)
      - Groq             (GROQ_API_KEY)
      - Ollama           (no key needed — but Ollama must be running)

    If .env is missing entirely, this step creates one interactively.
    """
    head("Step 4 · API key configuration")

    here    = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(here, ".env")
    ex_path  = os.path.join(here, ".env.example")

    if os.path.isfile(env_path):
        # File exists — check that at least one real key is present
        content = open(env_path).read()
        has_anthropic = "ANTHROPIC_API_KEY=" in content and \
                        "sk-ant-your-key-here" not in content and \
                        len([l for l in content.splitlines()
                             if l.startswith("ANTHROPIC_API_KEY=") and l.split("=",1)[1].strip()]) > 0
        has_groq      = "GROQ_API_KEY=" in content and \
                        len([l for l in content.splitlines()
                             if l.startswith("GROQ_API_KEY=") and l.split("=",1)[1].strip()]) > 0

        if has_anthropic or has_groq:
            ok(".env found with at least one API key")
            return
        else:
            warn(".env exists but no API keys are filled in.")
            warn("The app will only work if Ollama is running locally.")
            print()
            choice = input("  Would you like to add an Anthropic or Groq key now? (y/n): ").strip().lower()
            if choice == "y":
                _prompt_for_key(env_path)
            else:
                info("Continuing without API keys — make sure Ollama is running.")
    else:
        # No .env at all — create one
        warn(".env file not found.")
        print()
        print("  The app requires at least one of:")
        print("    • Anthropic API key  (https://console.anthropic.com)")
        print("    • Groq API key       (https://console.groq.com)")
        print("    • Ollama running locally (https://ollama.com) — no key needed")
        print()
        choice = input("  Would you like to enter an API key now? (y/n): ").strip().lower()

        # Start from the template if available, otherwise blank
        template = open(ex_path).read() if os.path.isfile(ex_path) else \
                   "ANTHROPIC_API_KEY=\nGROQ_API_KEY=\n"
        open(env_path, "w").write(template)

        if choice == "y":
            _prompt_for_key(env_path)
        else:
            info("Created blank .env — edit it later to add your API key.")
            info("Make sure Ollama is running if you have no cloud API key.")


def _prompt_for_key(env_path):
    """Interactive key entry helper."""
    print()
    print("  Which backend would you like to configure?")
    print("    1 · Anthropic Claude (recommended — best quality)")
    print("    2 · Groq (free tier available — fast)")
    print()
    choice = input("  Enter 1 or 2: ").strip()

    if choice == "1":
        key = input("  Paste your Anthropic API key (starts with sk-ant-): ").strip()
        _write_key(env_path, "ANTHROPIC_API_KEY", key)
    elif choice == "2":
        key = input("  Paste your Groq API key (starts with gsk_): ").strip()
        _write_key(env_path, "GROQ_API_KEY", key)
    else:
        warn("Skipping key entry.")


def _write_key(env_path, var_name, key):
    """Write or update a single key in .env."""
    if not key:
        warn("Empty key — skipping.")
        return

    content = open(env_path).read() if os.path.isfile(env_path) else ""
    lines   = content.splitlines()
    updated = False
    new_lines = []

    for line in lines:
        if line.startswith(f"{var_name}="):
            new_lines.append(f"{var_name}={key}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        new_lines.append(f"{var_name}={key}")

    open(env_path, "w").write("\n".join(new_lines) + "\n")
    ok(f"{var_name} saved to .env")


# ════════════════════════════════════════════════════════════════
# STEP 5 — Launch Streamlit
# ════════════════════════════════════════════════════════════════
def launch_app(python_path):
    """
    Start the Streamlit server using the venv's Python.

    Streamlit starts a local web server and opens a browser tab.
    The process keeps running until you press Ctrl+C in this terminal.
    """
    head("Step 5 · Launching the app")

    here     = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(here, "app.py")

    if not os.path.isfile(app_path):
        err("app.py not found. Make sure you are running run.py from the project folder.")
        input("\nPress Enter to exit.")
        sys.exit(1)

    sep()
    print()
    ok("Starting Feedback Analysis System …")
    info("The app will open in your browser at http://localhost:8501")
    info("To stop the app, press Ctrl+C in this window.")
    print()
    sep()

    # Small pause so the user can read the message before Streamlit output takes over
    time.sleep(1.5)

    # We exec Streamlit directly rather than calling `streamlit run` as a shell
    # command, because on Windows the PATH inside a subprocess may not resolve
    # `streamlit` even after pip install. Using the venv Python + -m guarantees
    # we use the correct installation.
    try:
        subprocess.run(
            [python_path, "-m", "streamlit", "run", app_path,
             "--server.headless", "false",
             "--browser.gatherUsageStats", "false"],
            cwd=here,
            check=False,   # Don't raise on Ctrl+C (returncode 1)
        )
    except KeyboardInterrupt:
        print()
        ok("App stopped. Goodbye!")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sep()
    print(_c("1", "  ◉  Feedback Analysis System — Launcher"))
    sep()

    try:
        check_python()
        python_path, pip_path = setup_venv()
        install_requirements(pip_path)
        check_env()
        launch_app(python_path)
    except KeyboardInterrupt:
        print()
        warn("Setup interrupted.")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        err(f"A setup step failed (exit code {e.returncode}).")
        err("Check the output above for details.")
        input("\nPress Enter to exit.")
        sys.exit(1)
    except Exception as e:
        err(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit.")
        sys.exit(1)
