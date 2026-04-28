---
title: LexUK Tenancy Chatbot
emoji: ⚖️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "5.49.0"
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: RAG + agentic chatbot over 12 UK tenancy statutes
---

# LexUK — UK Tenancy Law Chatbot

A retrieval-augmented chatbot that answers questions about UK residential tenancy law, grounded in 12 UK statutes.

## What it does

- Answers natural-language questions about UK tenancy law (assured shorthold tenancies, section 21 / section 8 notices, deposit protection, holding deposits, repair obligations, homelessness duties, etc.).
- Cites every legal claim inline using the format `[Act, s.X]`.
- Validates citations against the corpus (regex extraction + exact-section check).
- Refuses off-domain questions (medical, criminal, financial, general immigration) with a polite redirect.
- Two modes:
  - **Plain RAG** — direct retrieve → rerank → generate pipeline.
  - **Agentic RAG** — Planner agent on top, with tool-use orchestration (`search_statutes`, `validate_citations`, `refuse`, `clarify`).

## Architecture

| Stage | What it does |
|---|---|
| **Statute parser** | Deterministic regex + finite-state-machine over Markdown legislation. Produces canonical provision IDs (`s_21`, `sch_2_ground_8`). No LLM call. |
| **Embedding** | `text-embedding-3-large` over headline + summary + verbatim text. Indexed in ChromaDB with provision metadata. |
| **Retrieval** | Dual-query (original + LLM-rewritten) dense search PLUS exact-match metadata filter when the query names a specific provision. Exact-match hits are pinned to the generator's context. |
| **Reranking** | LLM reranker (Pydantic-typed structured output) reorders the top-K dense candidates. |
| **Generation** | `gpt-4o-mini` with a structured Act / Provision / Headline / Text context block; mandatory inline citations. |
| **Agent layer** | OpenAI tool-use loop on top — refusal of off-domain questions, post-hoc citation validation. |

## Corpus

12 UK Public General Acts (legislation.gov.uk, Open Government Licence v3.0):

- Rent Act 1977
- Protection from Eviction Act 1977
- Landlord and Tenant Act 1985
- Landlord and Tenant Act 1987
- Housing Act 1988
- Commonhold and Leasehold Reform Act 2002
- Housing Act 2004
- Immigration Act 2014
- Deregulation Act 2015
- Homelessness Reduction Act 2017
- Tenant Fees Act 2019
- Renters' Rights Act 2025

After deterministic parsing: ~2,300 provision-level chunks indexed.

## Performance (10-question gold-standard evaluation)

| Metric | Plain RAG | Agentic RAG |
|---|---:|---:|
| Expected-acts recall | 1.00 | 0.95 |
| MRR | 0.65 | 0.64 |
| NDCG | 0.69 | 0.67 |
| Keyword coverage | 0.80 | 0.77 |
| Citation validity | 1.00 | 1.00 |
| Refusal accuracy | 0.90 | **1.00** |
| Judge accuracy (1-5) | 4.00 | 4.22 |
| Judge completeness (1-5) | 4.12 | 3.56 |
| Judge relevance (1-5) | **4.88** | 4.44 |
| Latency p50 | 6.6 s | 8.1 s |

LLM-as-judge model: `gpt-4o` (chosen one tier above the system under test to mitigate self-preference bias, per Zheng et al. 2023).

## Running locally

```bash
git clone https://github.com/akifhameed/LexUK-Tenancy-Chatbot.git
cd LexUK-Tenancy-Chatbot

python -m venv .venv
.venv\Scripts\Activate.ps1            # Windows PowerShell
# source .venv/bin/activate           # macOS / Linux

pip install -r requirements.txt
```

Create a `.env` file with your OpenAI key:

```
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-large
LOG_LEVEL=INFO
```

Then launch:

```bash
python app.py                              # Gradio UI on http://127.0.0.1:7860
python -m scripts.chat_cli --agent         # CLI in agent mode
python -m eval.run_eval --config plain     # Run the eval
python -m eval.run_eval --config agent
```

## On Hugging Face Spaces

Deployed at: <https://huggingface.co/spaces/akifhameed/LexUK-Tenancy-Chatbot>

The Space requires `OPENAI_API_KEY` to be set as a Space Secret (Settings → Variables and secrets).

## Repository layout

```
app.py                         # Gradio UI entry point
requirements.txt
src/
├── config.py                  # Centralised settings + paths + RAG/AGENT tunables
├── llm_client.py              # OpenAI wrapper with tenacity retry
├── logging_setup.py
├── ingest/
│   ├── corpus_loader.py       # Load Markdown statutes + metadata
│   ├── statute_parser.py      # Deterministic provision parser (regex + FSM)
│   └── chunker.py             # Provision -> ChunkRecord, JSONL cache
├── rag/
│   ├── embeddings.py          # Batched OpenAI embeddings
│   ├── vector_store.py        # ChromaDB persistence + provision-aware filter
│   ├── query_rewriter.py
│   ├── retriever.py           # Dual-query dense + exact-match metadata filter
│   ├── reranker.py            # LLM reranker (Pydantic structured output)
│   ├── generator.py           # Answer generation with mandatory citations
│   └── pipeline.py            # End-to-end: rewrite -> retrieve -> rerank -> generate
└── agents/
    ├── base.py                # Agent base class
    ├── tools.py               # OpenAI function-calling tool schemas
    ├── statute_agent.py       # Wraps the RAG pipeline
    ├── citation_validator.py  # Pure-Python citation verifier
    ├── refusal_agent.py       # Templated off-domain refusals
    └── planner.py             # Tool-use loop orchestrator
eval/
├── dataset.py                 # Pydantic schema + JSONL loader
├── csv_to_jsonl.py            # Convert gold CSV to JSONL
├── metrics.py                 # MRR, NDCG, keyword coverage, citation validity
├── llm_judge.py               # gpt-4o as judge (accuracy/completeness/relevance)
├── run_eval.py                # Driver
└── inspect_results.py         # Per-row inspector
data/
├── markdown_12_statutes/      # Source statute markdowns (OGL v3.0)
├── chunks_cache/              # Per-statute JSONL chunk cache (deterministic)
└── chroma_db/                 # Persisted Chroma vector index
scripts/
├── build_chunks.py            # One-shot chunker
├── build_index.py             # Build / rebuild the Chroma index
├── chat_cli.py                # Terminal chat (plain or agent mode)
├── test_retrieval.py          # Smoke-test single queries
└── smoke_test.py              # Foundation sanity check
```

## Acknowledgements

UK statute text from <https://www.legislation.gov.uk> under the Open Government Licence v3.0.

## License

MIT — see [LICENSE](LICENSE).

This is information about UK statutes, **not legal advice**. For your specific situation consult a solicitor or Citizens Advice.
