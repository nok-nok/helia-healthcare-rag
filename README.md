# HeLiA Assistant 

## About
HeLiA (Healthcare Language AI Assistant) is a Retrieval-Augmented Generation (RAG) 
system that answers healthcare questions grounded in WHO and MedlinePlus documents. 
It uses hybrid retrieval (dense + BM25) and Mistral 7B via Ollama to generate 
cited, factual responses — reducing hallucination compared to prompting-only baselines.
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

## Setup Instructions

### Prerequisites
- Python 3.10 or 3.11 (not 3.12+)
- [Ollama](https://ollama.com/download) installed

### Step 1 — Download Mistral model
```bash
ollama pull mistral
```

### Step 2 — Clone the repository
```bash
git clone https://github.com/nok-nok/helia-healthcare-rag.git
cd helia-healthcare-rag
```

### Step 3 — Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 5 — Start Ollama server
```bash
ollama serve
```
> Keep this terminal open.

### Step 6 — Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> On first run, HeLiA automatically scrapes WHO and MedlinePlus pages,
> embeds all documents, and saves to ChromaDB (~2 minutes).
> Subsequent runs load instantly.
---

## Deployment
This application runs locally using Ollama + Mistral 7B.
Cloud deployment is not supported in this version as Ollama requires
a local environment. See the included installation guide for full
setup instructions.


