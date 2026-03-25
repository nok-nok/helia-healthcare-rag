# app.py
"""
Assignment 2 – HeLiA – Healthcare AI Assistant
Streamlit UI: Chat interface + Evaluation dashboard + Side-by-side comparison
"""

import sys
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ── Page config (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title="HeLiA – Healthcare AI Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.append(str(Path(__file__).parent))
from config import RETRIEVAL_K, LLM_MODEL
from utils.ingestion   import load_raw_documents, chunk_documents
from utils.vectorstore import VectorStore
from utils.rag_chain   import RAGChain


# ── Cache expensive resources ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_system():
    vs = VectorStore()
    if vs.is_empty():
        with st.spinner("📚 Building knowledge base (first run — takes ~2 min)..."):
            raw    = load_raw_documents("./docs")
            chunks = chunk_documents(raw)
            vs.index_chunks(chunks)
    chain = RAGChain(vs)
    return vs, chain


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    top_k     = st.slider("Retrieval top-k", 1, 10, RETRIEVAL_K)
    show_chunks = st.toggle("Show retrieved chunks", value=False)
    show_compare = st.toggle("Side-by-side baseline comparison", value=False)
    st.divider()
    st.markdown(f"**Model:** `{LLM_MODEL}` via Ollama")
    st.markdown("**Embeddings:** all-MiniLM-L6-v2 (local)")
    st.markdown("**Vector DB:** ChromaDB (persistent)")

    if st.button("🔄 Re-build Knowledge Base"):
        import shutil, os
        shutil.rmtree("./data/chroma_db", ignore_errors=True)
        st.cache_resource.clear()
        st.rerun()


# ── Load system ───────────────────────────────────────────────
vs, chain = load_system()

st.success(f"✅ Knowledge base ready — {vs.doc_count()} chunks indexed")


# ── Tabs ──────────────────────────────────────────────────────
tab_chat, tab_eval, tab_about = st.tabs(["💬 Chat", "📊 Evaluation", "ℹ️ About"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════
with tab_chat:
    st.header("🏥 HeLiA – Healthcare AI Assistant")
    st.caption("Ask any healthcare question. Answers are grounded in WHO & MedlinePlus documents.")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources"):
                    st.markdown(msg["sources"])
            if msg.get("chunks") and show_chunks:
                with st.expander(f"🔍 Retrieved {len(msg['chunks'])} chunks"):
                    for i, c in enumerate(msg["chunks"], 1):
                        st.markdown(f"**[{i}]** `{c['source']}` — score: `{c['score']:.3f}`")
                        st.text(c["text"][:300] + "...")
                        st.divider()

    # Input
    if prompt := st.chat_input("Ask a healthcare question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if show_compare:
                col_rag, col_base = st.columns(2)

                with col_rag:
                    st.markdown("**🤖 RAG Answer**")
                    with st.spinner("Retrieving & generating…"):
                        result = chain.answer(prompt, k=top_k)
                    st.markdown(result["answer"])
                    with st.expander("📎 Sources"):
                        st.markdown(result["sources_md"])

                with col_base:
                    st.markdown("**⚡ Baseline (no RAG)**")
                    with st.spinner("Generating baseline…"):
                        base_ans = chain.answer_baseline(prompt)
                    st.markdown(base_ans)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**RAG:** {result['answer']}\n\n---\n**Baseline:** {base_ans}",
                    "sources": result["sources_md"],
                    "chunks": result["chunks"],
                })

            else:
                with st.spinner("Searching knowledge base…"):
                    result = chain.answer(prompt, k=top_k)
                st.markdown(result["answer"])

                if result["sources_md"]:
                    with st.expander("📎 Sources"):
                        st.markdown(result["sources_md"])

                if show_chunks and result["chunks"]:
                    with st.expander(f"🔍 Retrieved {len(result['chunks'])} chunks"):
                        for i, c in enumerate(result["chunks"], 1):
                            st.markdown(f"**[{i}]** `{c['source']}` — score: `{c['score']:.3f}`")
                            st.text(c["text"][:300] + "...")
                            st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources_md"],
                    "chunks": result["chunks"],
                })

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB 2 — EVALUATION
# ══════════════════════════════════════════════════════════════
with tab_eval:
    st.header("📊 Evaluation Dashboard")
    st.caption("Compare A1 Baseline vs A2 RAG on 25 test cases.")

    results_csv = Path("./evaluation/results/evaluation_results.csv")

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("▶️ Run Full Evaluation (25 questions)", type="primary"):
            from evaluation.evaluate import run_evaluation, generate_all_plots
            progress = st.progress(0, text="Starting evaluation…")
            with st.spinner("Running evaluation — this takes several minutes…"):
                df = run_evaluation(chain)
                generate_all_plots(df)
            st.success("✅ Evaluation complete!")
            st.rerun()

    with col2:
        if results_csv.exists():
            with open(results_csv, "rb") as f:
                st.download_button("⬇️ Download CSV", f, "evaluation_results.csv", "text/csv")

    if results_csv.exists():
        df = pd.read_csv(results_csv)

        # Summary metrics
        st.subheader("Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RAG Helpfulness",  f"{df['rag_helpfulness'].mean():.2f}",
                  f"+{df['rag_helpfulness'].mean() - df['base_helpfulness'].mean():.2f} vs baseline")
        m2.metric("RAG Safety",       f"{df['rag_safety'].mean():.2f}")
        m3.metric("Groundedness",     f"{df['rag_groundedness'].mean():.2f}")
        m4.metric("Citation Accuracy",f"{df['rag_citation_accuracy'].mean():.2f}")

        # Visualizations
        viz_dir = Path("./evaluation/results")
        st.subheader("Visualizations")

        v1, v2 = st.columns(2)
        if (viz_dir / "viz1_before_after.png").exists():
            with v1:
                st.image(str(viz_dir / "viz1_before_after.png"), caption="Before vs After", use_container_width=True)
        if (viz_dir / "viz2_by_category.png").exists():
            with v2:
                st.image(str(viz_dir / "viz2_by_category.png"), caption="By Category", use_container_width=True)
        if (viz_dir / "viz3_rag_metrics.png").exists():
            st.image(str(viz_dir / "viz3_rag_metrics.png"), caption="RAG Metric Distributions", use_container_width=True)

        # Full results table
        st.subheader("Full Results Table")
        display_cols = ["id", "category", "question",
                        "base_helpfulness", "base_safety", "base_clarity",
                        "rag_helpfulness",  "rag_safety",  "rag_clarity",
                        "rag_groundedness", "rag_citation_accuracy"]
        st.dataframe(df[display_cols], use_container_width=True)

    else:
        st.info("No evaluation results yet. Click 'Run Full Evaluation' above.")


# ══════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════
with tab_about:
    st.header("ℹ️ About This Application")
    st.markdown("""
## CMPE 682/683 – Assignment 2: HeLiA – Healthcare AI Assistant

### Architecture
```
Knowledge Base (WHO + MedlinePlus)
        ↓ load_raw_documents()
Text Chunking (RecursiveCharacterTextSplitter, 400 tokens, 80 overlap)
        ↓ chunk_documents()
Embeddings (all-MiniLM-L6-v2, local)
        ↓ VectorStore.index_chunks()
ChromaDB (persistent vector database)
        ↓ VectorStore.retrieve()
Hybrid Retrieval (Dense cosine + BM25, α=0.7)
        ↓ RAGChain.answer()
Mistral via Ollama (grounded generation with citations)
```

### Knowledge Base
- **WHO Fact Sheets**: Diabetes, Hypertension, Asthma, Depression, Obesity, Cancer, Cardiovascular
- **MedlinePlus**: High Blood Pressure, Type 2 Diabetes, Heart Disease
- **Local docs**: Place PDFs/TXTs in `./docs/` folder

### Key Design Decisions
| Choice | Rationale |
|---|---|
| Mistral via Ollama | Free, local, no API cost |
| all-MiniLM-L6-v2 | Fast, small, good quality, no API key |
| ChromaDB | Simple persistent storage, no server needed |
| Hybrid retrieval | Better recall than dense-only |
| 400-token chunks | Balances context richness vs retrieval precision |

### Track A Deliverables
- ✅ Document collection and processing (Component 1)
- ✅ Embedding and retrieval with hybrid BM25+dense (Component 2)
- ✅ Grounded generation with citation system (Component 3)
- ✅ 25-question evaluation dataset
- ✅ 3 visualizations
- ✅ Streamlit deployment
    """)
