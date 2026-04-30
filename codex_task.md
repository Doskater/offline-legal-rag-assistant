# Task Brief: Offline Legal RAG Assistant Portfolio Case

Build a local portfolio project inspired by a client request for an offline legal research and drafting assistant on macOS.

The project must demonstrate:

- Local-only document ingestion.
- Resumable indexing with SQLite checkpoints.
- Hybrid retrieval using BM25 plus vector similarity.
- Source-grounded answers with numbered citations.
- Exact source file and passage references.
- A style/preferences file that affects drafting tone.
- An evaluation script with 10 test queries.
- A handover document explaining install, run, re-indexing, and troubleshooting.

Keep the demo safe and self-contained:

- Do not use real legal client documents.
- Use synthetic public-style legal/business demo documents.
- Do not require cloud APIs or real API keys.
- Make Ollama optional; the demo should run with deterministic local fallback embeddings if Ollama is not installed.

Required outputs:

- `sample_output/daily_report.md`
- `sample_output/ranked_queries.csv`
- `sample_output/index_stats.json`
- `sample_output/sample_answer.md`
- `CLIENT_HANDOFF.md`
- `README.md`
- `proposal_draft.md`

