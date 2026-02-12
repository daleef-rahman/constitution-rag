# Constitution RAG: Comparing Retrieval Strategies on the Indian Constitution

A systematic comparison of RAG (Retrieval-Augmented Generation) pipelines applied to the Indian Constitution (404 pages, 877K+ characters). Each pipeline is evaluated against a hand-crafted 35-question benchmark spanning five difficulty tiers.

## RAG Pipelines

All pipelines share a common base class (`RAGPipeline`) and use **text-embedding-3-small** for embeddings, **ChromaDB** for vector storage, and **Llama-3.2-3B-Instruct** (via Together AI) for generation.

| Pipeline | Chunking | Retrieval Strategy |
|---|---|---|
| **No RAG** | — | None — LLM answers from parametric knowledge only |
| **Naive RAG** | Fixed-size (1000 chars, 200 overlap) | Cosine similarity over embeddings |
| **Reranker RAG** | Fixed-size | Vector search → LLM relevance scoring to rerank top-k |
| **Hybrid Search RAG** | Fixed-size | BM25 + semantic search fused via Reciprocal Rank Fusion |
| **Hierarchical RAG** | Two-level (5000-char sections + 1000-char chunks) | LLM-summarised sections → filtered detail retrieval |

## Evaluation

Uses an **LLM-as-a-Judge** approach (GPT-4o) that scores each response against ground truth as PASS (1), PARTIAL (0.5), or FAIL (0).

The 35 validation questions are grouped into five tiers:
- **Tier 1 — Direct Retrieval**: Single-article factual lookups
- **Tier 2 — Multi-Hop Reasoning**: Questions requiring linked reasoning across articles
- **Tier 3 — Comparative Analysis**: Side-by-side comparison of constitutional provisions
- **Tier 4 — Scenario/Application**: Applying articles to hypothetical situations
- **Tier 5 — Negative Constraints**: Questions where the correct answer is "no" or "not mentioned"
- **Global Understanding**: Thematic synthesis across the entire document

## Setup

### Prerequisites
- Python 3.10+
- [Together AI](https://together.ai/) API key (for Llama-3.2 inference)
- [OpenAI](https://platform.openai.com/) API key (for embeddings + GPT-4o evaluation)

### Install

```bash
pip install pymupdf chromadb openai rank-bm25 python-dotenv
```

### Environment Variables

Create a `.env` file in the parent directory:

```
OPENAI_API_KEY=sk-...
TOGETHER_API_KEY=...
```

### Run

Open `rag.ipynb` and run cells top to bottom. Switch the active pipeline by changing the `Pipeline = ...` assignment before the evaluation cells.

## Project Structure

```
├── rag.ipynb                # Main notebook — all pipelines + evaluation
├── val.json                 # 35-question validation benchmark
├── indiaconstitution.pdf    # Source document (Constitution of India, May 2022)
└── README.md
```
