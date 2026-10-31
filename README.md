# NexusRAG 📚

> **Production-Grade Enterprise Retrieval-Augmented Generation Engine**  
> *Hierarchical (Parent-Child) Chunking · Hybrid Search (Dense + BM25) · Reciprocal Rank Fusion · Cross-Encoder Reranking · Automated Evaluation Benchmarks*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3+-1C3C3C?style=flat)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.50+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)

---

## 🚀 Overview

Standard RAG systems rely on a naive retrieve-and-generate loop: a single bi-encoder embeds queries and uniform text chunks, runs cosine similarity, and dumps raw text into an LLM context. In real-world enterprise environments, this fails due to:
1. **Vocabulary & Keyword Blindspots:** Dense embeddings frequently miss exact acronyms, policy IDs, and product codes.
2. **The Chunking Dilemma:** Small chunks produce sharp vector embeddings but lack context; large chunks preserve context but dilute vector semantic density.
3. **Retrieval Noise:** Cosine similarity does not equal relevance. Irrelevant chunks trigger hallucinations or "lost-in-the-middle" attention degradation.

**NexusRAG** is engineered to eliminate these failure modes using a multi-stage retrieval, reranking, and refinement architecture.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["1. Ingestion & Hierarchical Chunking"]
        Docs["Raw Documents (PDF, DOCX, TXT)"] --> Clean["Text Cleaning & Section Extraction"]
        Clean --> ParentSplit["Parent Chunks (1600 chars, overlap 200)"]
        ParentSplit --> ChildSplit["Child Chunks (400 chars, overlap 60)"]
        ParentSplit --> DocStore[("DocStore (doc_store.json)")]
        ChildSplit --> Embeddings["Gemini Embedding Model"]
        Embeddings --> VectorDB[("Vector DB (FAISS / Pinecone)")]
        ChildSplit --> BM25Index[("BM25 Sparse Index")]
    end

    subgraph QueryPipeline ["2. Multi-Stage Retrieval & Refinement"]
        UserQuery["User Question + Conversation History"] --> MultiQuery["Mistral Multi-Query Rewrite (3 variants)"]
        MultiQuery --> DenseSearch["Dense Vector Search (Top-6)"]
        MultiQuery --> SparseSearch["BM25 Keyword Search (Top-6)"]
        DenseSearch & SparseSearch --> RRF["Reciprocal Rank Fusion (Top-10)"]
        RRF --> ParentRecovery["Parent Recovery (Child -> Parent Lookup)"]
        ParentRecovery --> Rerank["Cross-Encoder Reranking (ms-marco-MiniLM)"]
        Rerank --> Refinement["Threshold Filter (-4.0) & Budgeting (8000 chars)"]
    end

    subgraph Generation ["3. Synthesis & Grounding"]
        Refinement --> Prompt["Grounded Prompt + Formatted Sources"]
        Prompt --> LLM["Mistral Chat LLM (Temp: 0.0)"]
        LLM --> Response["Grounded Answer + Citations [1], [2]"]
        Response --> Memory["Sliding Window Memory (4 turns)"]
    end
```

---

## ✨ Key Features

* **Hierarchical (Parent-Child) Chunking:**
  * Decouples the **search unit** (400-char child chunks) from the **synthesis unit** (1,600-char parent chunks).
  * Child vectors capture granular semantics without dilution; parent chunks supply full paragraphs and surrounding narrative to the generator LLM.
* **Hybrid Search (Dense + Sparse):**
  * Simultaneously queries **Dense Semantic Vectors** (Google Gemini embeddings via FAISS/Pinecone) and **Sparse Lexical Inverted Indexes** (BM25).
* **Reciprocal Rank Fusion (RRF, $k=60$):**
  * Monotonically merges 8 ranked candidate lists (dense & sparse lists across 4 query variants) without brittle score calibration.
* **Conversational Co-reference Resolution:**
  * LLM-driven query rewriting resolves pronouns ("it", "that policy", "the second tier") using a sliding conversation history window (4 turns / 8 messages).
* **Cross-Encoder Reranking & Fallback:**
  * Second-stage deep transformer reranking (`ms-marco-MiniLM-L-6-v2`) with an automated regex-based keyword density fallback for CPU-only or restricted runtimes.
* **Context Refinement & Guardrails:**
  * Discards low-confidence chunks ($\text{logit} < -4.0$) and enforces a hard **8,000-character context budget** to prevent context-stuffing.
* **Interactive 5-Stage Debug Inspector:**
  * Full observability panel in the Streamlit UI displaying original query, rewritten variants, retrieved candidates, reranked scores, and the final LLM prompt context.

---

## 📊 Evaluation & Benchmark Results

NexusRAG includes an automated quantitative evaluation harness tested against a gold-standard enterprise policy test set (`data/evaluation/ground_truth.json`):

| Metric | Benchmark Result | Description |
| :--- | :---: | :--- |
| **Retrieval Recall@K** | **100.0%** | Target document successfully retrieved in candidate list |
| **Context Hit Rate** | **100.0%** | Relevant document survived reranking into final context |
| **Mean Reciprocal Rank (MRR)** | **1.000** | Ground-truth chunk consistently ranked at position #1 |
| **Key Phrase Coverage** | **85.0%** | Crucial factual details present in generated synthesis |
| **Average Query Latency** | **7.48s** | End-to-end execution including LLM rewrite, search & synthesis |

---

## 📂 Project Structure

```text
NexusRAG/
├── data/
│   ├── documents/             # Raw ingestion files (PDF, DOCX, TXT)
│   └── evaluation/
│       ├── ground_truth.json  # Benchmark dataset (10 test cases)
│       └── eval_results.json  # Output metrics from evaluation runs
├── scripts/
│   ├── build_index.py         # Full vector & document store indexing script
│   ├── update_index.py        # Incremental indexing script for new files
│   ├── run_evals.py           # Quantitative evaluation harness
│   └── generate_sample_documents.py # Synthetic sample doc generator
├── src/
│   ├── chunking.py            # Hierarchical parent-child splitter
│   ├── indexing.py            # FAISS / Pinecone store lifecycle management
│   ├── ingestion.py           # Document parsers, regex cleaner & section extractor
│   ├── memory.py              # Sliding window chat history & context trimmer
│   ├── rag_pipeline.py        # Orchestration pipeline from retrieval to answer
│   ├── refinement.py          # Logit filtering, context budgeting & citations
│   ├── reranking.py           # Cross-encoder reranker with keyword fallback
│   └── retrieval.py           # Multi-query rewriting, BM25, and RRF fusion
├── vectorstore/
│   ├── doc_store.json         # Serialized parent & child document lookup
│   └── faiss/                 # Local FAISS index files (index.faiss, index.pkl)
├── app.py                     # Streamlit frontend & chat interface
├── config.py                  # Centralized configuration & environment loader
├── llm_client.py              # Cached factory for Mistral and Gemini models
├── Dockerfile                 # Production container setup (Python 3.11-slim)
├── .dockerignore              # Docker build exclusion rules
└── requirements.txt           # Project dependencies
```

---

## ⚡ Quickstart Guide

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/your-username/NexusRAG.git
cd NexusRAG

# Create virtual environment
python -m venv nexus_venv

# Activate virtual environment
# Windows:
.\nexus_venv\Scripts\activate
# Linux/macOS:
source nexus_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
# LLM & Embedding API Keys
MISTRAL_API_KEY=your_mistral_api_key_here
MISTRAL_CHAT_MODEL=mistral-small-latest
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_EMBEDDING_MODEL=models/embedding-001

# Vector Store Selection: "faiss" or "pinecone"
VECTOR_DB=faiss

# Optional Pinecone Configuration (only if VECTOR_DB=pinecone)
PINECONE_API_KEY=
PINECONE_INDEX_NAME=rag
PINECONE_NAMESPACE=NexusRAG

# Pipeline Parameters
PARENT_CHUNK_SIZE=1600
PARENT_CHUNK_OVERLAP=200
CHILD_CHUNK_SIZE=400
CHILD_CHUNK_OVERLAP=60
NUM_QUERY_VARIANTS=3
VECTOR_TOP_K=6
BM25_TOP_K=6
HYBRID_TOP_K=10
RERANK_TOP_K=4
MIN_RERANK_SCORE=-4.0
MAX_CONTEXT_CHARS=8000
MEMORY_WINDOW=4
RERANKER=cross-encoder
```

### 3. Build the Index

Place your PDF, DOCX, or TXT documents into `data/documents/`, then run:

```bash
python scripts/build_index.py
```

### 4. Launch the Streamlit App

```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 🧪 Running Evaluations

Run the automated evaluation benchmark suite:

```bash
# 1. Full End-to-End Evaluation (measures LLM generation & factual accuracy)
python scripts/run_evals.py

# 2. Offline Retrieval & Reranking Evaluation (no LLM generation API calls)
python scripts/run_evals.py --retrieval-only
```
Results and per-query logs are exported to `data/evaluation/eval_results.json`.

---

## 🐳 Docker Deployment

You can run the entire system inside a self-contained Docker container:

### Build Image
```bash
docker build -t nexusrag:latest .
```

### Run Container
```bash
# Run with local volume persistence for uploaded data and indices
docker run -p 8501:8501 \
  --env-file .env \
  -v "${PWD}/data:/app/data" \
  -v "${PWD}/vectorstore:/app/vectorstore" \
  nexusrag:latest
```
Access the application at **http://localhost:8501**.

---