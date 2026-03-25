# utils/ingestion.py
"""
Component 1: Document Collection and Processing
- Loads PDFs and plain-text files from the /docs folder
- Scrapes WHO / MedlinePlus pages
- Chunks with overlap, stores metadata (source, page, section)
"""

import os
import re
import requests
from pathlib import Path
from typing import List, Dict

from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP

# ── WHO fact-sheet URLs (edit / add as needed) ────────────────
WHO_URLS = [
    ("Diabetes",        "https://www.who.int/news-room/fact-sheets/detail/diabetes"),
    ("Hypertension",    "https://www.who.int/news-room/fact-sheets/detail/hypertension"),
    ("Asthma",          "https://www.who.int/news-room/fact-sheets/detail/asthma"),
    ("Depression",      "https://www.who.int/news-room/fact-sheets/detail/depression"),
    ("Obesity",         "https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight"),
    ("Cancer",          "https://www.who.int/news-room/fact-sheets/detail/cancer"),
    ("Cardiovascular",  "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)"),
]

MEDLINEPLUS_URLS = [
    ("High Blood Pressure", "https://medlineplus.gov/highbloodpressure.html"),
    ("Type 2 Diabetes",     "https://medlineplus.gov/diabetestype2.html"),
    ("Heart Disease",       "https://medlineplus.gov/heartdisease.html"),
]


def _scrape_url(url: str, title: str) -> Dict:
    """Fetch and clean text from a URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research bot for academic assignment)"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # Remove nav/script/style noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return {"text": text, "source": title, "url": url, "type": "web"}
    except Exception as e:
        print(f"  [WARN] Could not fetch {url}: {e}")
        return None


def _load_pdf(path: str) -> List[Dict]:
    """Load pages from a PDF file."""
    docs = []
    try:
        reader = PdfReader(path)
        name = Path(path).stem
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append({
                    "text": text,
                    "source": name,
                    "url": path,
                    "type": "pdf",
                    "page": i + 1,
                })
    except Exception as e:
        print(f"  [WARN] Could not read PDF {path}: {e}")
    return docs


def _load_txt(path: str) -> Dict:
    """Load a plain-text file."""
    name = Path(path).stem
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return {"text": text, "source": name, "url": path, "type": "txt"}


def load_raw_documents(docs_dir: str = "./docs") -> List[Dict]:
    """Collect raw documents from web + local files."""
    raw = []

    # 1. WHO fact sheets
    print("Fetching WHO fact sheets...")
    for title, url in WHO_URLS:
        doc = _scrape_url(url, f"WHO – {title}")
        if doc:
            raw.append(doc)
            print(f"  ✓ {title}")

    # 2. MedlinePlus
    print("Fetching MedlinePlus pages...")
    for title, url in MEDLINEPLUS_URLS:
        doc = _scrape_url(url, f"MedlinePlus – {title}")
        if doc:
            raw.append(doc)
            print(f"  ✓ {title}")

    # 3. Local PDFs / TXTs
    docs_path = Path(docs_dir)
    if docs_path.exists():
        for f in docs_path.glob("*.pdf"):
            print(f"  Loading PDF: {f.name}")
            raw.extend(_load_pdf(str(f)))
        for f in docs_path.glob("*.txt"):
            print(f"  Loading TXT: {f.name}")
            raw.append(_load_txt(str(f)))

    print(f"\nTotal raw documents collected: {len(raw)}")
    return raw


def chunk_documents(raw_docs: List[Dict]) -> List[Dict]:
    """
    Split raw documents into overlapping chunks.
    Returns a list of chunk dicts with text + metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in raw_docs:
        splits = splitter.split_text(doc["text"])
        for i, split in enumerate(splits):
            chunk = {
                "text": split.strip(),
                "source": doc.get("source", "Unknown"),
                "url":    doc.get("url", ""),
                "type":   doc.get("type", ""),
                "chunk_index": i,
                "page":   doc.get("page", None),
            }
            chunks.append(chunk)

    print(f"Total chunks after splitting: {len(chunks)}")
    return chunks
