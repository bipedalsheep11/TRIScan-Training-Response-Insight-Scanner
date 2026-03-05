# backend/utils/document_reader.py
# ─────────────────────────────────────────────────────────────────
# Unified document loader. Supports PDF, DOCX, CSV, and XLSX.
# Returns a dict with 'raw_text' and/or 'dataframe' depending
# on the file type.
# ─────────────────────────────────────────────────────────────────

from pathlib import Path
import pandas as pd
import io


def extract_text_from_pdf(file) -> str:
    try:
        import fitz  # PyMuPDF — pip install pymupdf
    except ImportError:
        raise ImportError(
            "PyMuPDF is required to read PDF files.\n"
            "Install it with:  pip install pymupdf"
        )

    # Ensure the file pointer is at the start.
    # .seek(0) moves to byte 0 (the beginning) of the file-like object.
    file.seek(0)

    # fitz.open() with stream= and filetype= accepts a BytesIO object.
    # It reads the raw bytes and parses the PDF structure.
    doc = fitz.open(stream=file.read(), filetype="pdf")

    pages_text = []
    for page_number, page in enumerate(doc, start=1):
        # .get_text("text") extracts the text in reading order (left → right, top → bottom).
        page_text = page.get_text("text")
        if page_text.strip():  # Skip blank or image-only pages.
            pages_text.append(f"[Page {page_number}]\n{page_text}")

    doc.close()

    # "\n\n".join() puts a blank line between each page block.
    return "\n\n".join(pages_text)


def extract_text_from_docx(file) -> str:
    """
    Extract all text from a Word .docx file, preserving headings.
    """
    try:
        from docx import Document
        doc   = Document(file)
        parts = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            if para.style.name.startswith("Heading"):
                parts.append(f"\n## {text}")
            else:
                parts.append(text)
        return "\n".join(parts)
    except ImportError:
        return "[python-docx not installed — install with: pip install python-docx]"
    except Exception as e:
        return f"[DOCX extraction error: {e}]"


def load_file(file, filename: str) -> dict:
    """
    Unified entry point — detects file type from extension and
    routes to the correct extractor.

    Parameters
    ----------
    file     : file-like object (from Streamlit st.file_uploader)
    filename : str — original filename, used to detect extension

    Returns
    -------
    dict with keys:
      'raw_text'   : str | None
      'dataframe'  : pd.DataFrame | None
      'file_type'  : str — 'pdf', 'docx', 'csv', 'xlsx'
      'page_count' : int | None (PDF only)
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        text = extract_text_from_pdf(file)
        return {
            "raw_text":   text,
            "dataframe":  None,
            "file_type":  "pdf",
            "page_count": text.count("[Page "),
        }

    elif ext == ".docx":
        text = extract_text_from_docx(file)
        return {
            "raw_text":   text,
            "dataframe":  None,
            "file_type":  "docx",
            "page_count": None,
        }

    elif ext == ".csv":
        try:
            df = pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding="latin-1")
        return {
            "raw_text":   None,
            "dataframe":  df,
            "file_type":  "csv",
            "page_count": None,
        }

    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file)
        return {
            "raw_text":   None,
            "dataframe":  df,
            "file_type":  "xlsx",
            "page_count": None,
        }

    elif ext == ".txt":
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        return {
            "raw_text":   content,
            "dataframe":  None,
            "file_type":  "txt",
            "page_count": None,
        }

    else:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            "Supported formats: PDF, DOCX, TXT, CSV, XLSX"
        )
