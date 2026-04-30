# Offline Legal RAG Assistant

A local portfolio case study for a private legal research and drafting assistant that runs without cloud APIs. The demo indexes a safe synthetic legal/business corpus, performs hybrid retrieval, and produces source-grounded answers with numbered citations.

## Business Problem

Legal researchers often need to search private document collections without uploading files to cloud services. The workflow also needs verifiable citations: every answer should point back to the source file and passage used.

## Solution

This project demonstrates a local RAG workflow:

1. Ingest local Markdown, text, and HTML files.
2. Split documents into overlapping chunks.
3. Store indexing state in SQLite so failed or changed files can be resumed.
4. Use BM25 plus vector similarity for hybrid retrieval.
5. Generate grounded answers with numbered footnotes and passage references.
6. Run a 10-query evaluation to check whether retrieval finds the expected source material.

The demo uses deterministic local hash embeddings by default so it runs without any model downloads. It also supports an optional Ollama embedding path for real local deployments.

## Features

- Local-only corpus ingestion
- SQLite document and chunk index
- File hash tracking for resumable indexing
- BM25 keyword retrieval
- Vector retrieval with local fallback embeddings
- Optional Ollama `/api/embed` embeddings
- Hybrid score weighting
- Numbered source citations
- Exact source file and chunk references
- Style/preferences file
- 10-query evaluation report
- Plain-English client handoff document

## Tech Stack

- Python 3
- SQLite
- Standard-library HTML parser
- BM25 scoring
- Local deterministic embeddings
- Optional Ollama embeddings

## Project Structure

```text
offline-legal-rag-assistant/
├── rag_assistant.py
├── sample_corpus/
├── sample_queries.csv
├── preferences.md
├── sample_output/
├── CLIENT_HANDOFF.md
├── proposal_draft.md
├── requirements.txt
└── README.md
```

## How To Run

From this project folder:

```bash
python3 rag_assistant.py demo
```

The demo writes:

```text
sample_output/index_stats.json
sample_output/ranked_queries.csv
sample_output/daily_report.md
sample_output/sample_answer.md
```

Ask a single question:

```bash
python3 rag_assistant.py ask "What are the document retention rules for client matter files?"
```

Re-run only indexing:

```bash
python3 rag_assistant.py index
```

Run only evaluation:

```bash
python3 rag_assistant.py eval
```

## Optional Ollama Mode

Install Ollama and pull an embedding model:

```bash
ollama pull nomic-embed-text
```

Then run:

```bash
python3 rag_assistant.py demo --use-ollama --embedding-model nomic-embed-text
```

For a Legal X-style client setup, the full model list would be pulled separately:

```bash
ollama pull qwen2.5:14b
ollama pull mistral
ollama pull nomic-embed-text
ollama pull deepseek-r1:14b
```

## Example Output

The demo evaluation currently tests 10 queries and writes the pass/fail result to:

```text
sample_output/daily_report.md
```

The sample answer includes:

- short answer
- style note
- numbered footnotes
- source file path
- chunk number
- source passage
- hybrid, BM25, and vector scores

## How This Maps To A Client RAG Setup

This portfolio case is intentionally smaller than a production legal corpus. The same architecture scales into a client discovery phase:

1. Install Ollama on Apple Silicon.
2. Confirm local model execution and GPU/Metal usage.
3. Set up Kotaemon, AnythingLLM, LlamaIndex, or a custom local RAG stack.
4. Index a 50-document sample first.
5. Validate citations and source highlighting.
6. Tune chunk size, top-K, and hybrid weights.
7. Run a fixed evaluation set before indexing the full corpus.
8. Use checkpoints so failed embedding steps can resume without starting from zero.

## Portfolio Positioning

Use this project for jobs involving:

- local RAG
- private document search
- legal research assistants
- Ollama setup
- offline AI workflows
- citation-grounded answers
- resumable indexing
- BM25 plus vector retrieval

## Limitations

- This demo does not parse binary PDF files without optional dependencies.
- The default answer generator is extractive and deterministic, not a full LLM drafting system.
- Source highlighting is represented by exact chunk passage references; a production Kotaemon setup would handle UI-level document highlighting.
- This is a portfolio-safe prototype, not legal advice software.

