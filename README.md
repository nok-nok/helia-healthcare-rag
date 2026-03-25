# HeLiA Assistant — Assignment 2

**CMPE 682/683/782/783 · Qatar University · Spring 2026**  
**Track A: Retrieval-Augmented Generation**

---

## Architecture

```
Knowledge Base Documents (WHO + MedlinePlus + local PDFs)
        ↓  load_raw_documents()          [utils/ingestion.py]
Text Chunking  (400 tokens, 80 overlap)
        ↓  chunk_documents()
Embeddings  (all-MiniLM-L6-v2, local)
        ↓  VectorStore.index_chunks()    [utils/vectorstore.py]
ChromaDB  (persistent vector database)
        ↓  VectorStore.retrieve()
Hybrid Retrieval  (Dense cosine + BM25, α=0.7)
        ↓  RAGChain.answer()             [utils/rag_chain.py]
Mistral via Ollama  (grounded response + citations)
```

---

## Setup

### 1. Prerequisites

```bash
# Install Ollama  (https://ollama.com)
ollama pull mistral

# Verify it works
ollama run mistral "hello"
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Add your own documents

Put PDFs or `.txt` files in the `./docs/` folder — they will be ingested automatically.

Suggested sources:
- **WHO**: https://www.who.int/news-room/fact-sheets
- **MedlinePlus**: https://medlineplus.gov/
- **FDA drug labels**: https://labels.fda.gov/

### 4. Build the knowledge base (optional, done automatically on first run)

```bash
python setup_kb.py
```

### 5. Launch the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Deployment (Streamlit Cloud)

1. Push this repo to GitHub
2. Go to https://streamlit.io/cloud → New App → select `app.py`
3. **Note:** Ollama cannot run on Streamlit Cloud. For cloud deployment, swap
   `_call_ollama()` in `utils/rag_chain.py` with the OpenAI or Groq API.
   Set your API key as a Streamlit secret.

---

## Project Structure

```
healthcare_rag/
├── app.py                  # Streamlit UI
├── config.py               # All settings (edit here)
├── setup_kb.py             # One-time KB build script
├── requirements.txt
├── docs/                   # Place your PDFs/TXTs here
├── data/
│   └── chroma_db/          # Auto-created vector database
├── utils/
│   ├── ingestion.py        # Component 1: doc loading + chunking
│   ├── vectorstore.py      # Component 2: embeddings + hybrid retrieval
│   └── rag_chain.py        # Component 3: grounded generation
└── evaluation/
    ├── evaluate.py         # 25-question test suite + visualizations
    └── results/            # Auto-created after running evaluation
```

---

## Configuration (config.py)

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `mistral` | Any Ollama model |
| `CHUNK_SIZE` | `400` | Tokens per chunk |
| `CHUNK_OVERLAP` | `80` | Overlap between chunks |
| `RETRIEVAL_K` | `5` | Top-k chunks to retrieve |
| `USE_HYBRID` | `True` | Dense + BM25 hybrid |
| `HYBRID_ALPHA` | `0.7` | 0=BM25 only, 1=dense only |

---

## Evaluation

Run from the **Evaluation** tab in the app, or directly:

```python
from utils.rag_chain import RAGChain
from utils.vectorstore import VectorStore
from evaluation.evaluate import run_evaluation, generate_all_plots

vs    = VectorStore()
chain = RAGChain(vs)
df    = run_evaluation(chain)
generate_all_plots(df)
```

Results saved to `./evaluation/results/`.

---

## Team Contributions

| Member | Contribution |
|---|---|
| [Name 1] | Document collection, ingestion pipeline, evaluation |
| [Name 2] | Vector store, RAG chain, Streamlit UI |

---

## References

- Huyen, C. (2024). *AI Engineering*, Chapter 6.
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- LangChain documentation: https://python.langchain.com
- ChromaDB documentation: https://docs.trychroma.com
- WHO Fact Sheets: https://www.who.int/news-room/fact-sheets
- MedlinePlus: https://medlineplus.gov/
