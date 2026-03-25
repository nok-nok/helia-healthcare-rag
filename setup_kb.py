#!/usr/bin/env python3
# setup_kb.py
"""
One-time script to build the knowledge base.
Run this BEFORE launching the Streamlit app if you want
to pre-index documents without waiting in the UI.

Usage:
    python setup_kb.py
"""

from utils.ingestion   import load_raw_documents, chunk_documents
from utils.vectorstore import VectorStore

if __name__ == "__main__":
    print("=" * 55)
    print("  Healthcare RAG — Knowledge Base Setup")
    print("=" * 55)

    vs = VectorStore()
    if not vs.is_empty():
        print(f"Already indexed {vs.doc_count()} chunks.")
        ans = input("Re-index? (y/N): ").strip().lower()
        if ans != "y":
            print("Skipping.")
            exit(0)
        import shutil
        shutil.rmtree("./data/chroma_db", ignore_errors=True)
        vs = VectorStore()

    raw    = load_raw_documents("./docs")
    chunks = chunk_documents(raw)
    vs.index_chunks(chunks)

    print(f"\n✓ Knowledge base ready with {vs.doc_count()} chunks.")
    print("  Run the app with:  streamlit run app.py")
