# Proposal Draft: Offline Legal RAG / Ollama Setup

Hi,

I can help with a local-first RAG setup and would start with a controlled sanity phase before indexing the full corpus.

Relevant prior work: I built a local offline RAG assistant prototype using Python, SQLite, BM25 + vector hybrid retrieval, resumable document indexing, source-file citations, passage-level references, and a 10-query evaluation harness. The demo is designed specifically around private legal-style document search without cloud APIs.

My plan would be:

1. Confirm Ollama is installed and running locally.
2. Pull and test the requested models.
3. Verify Apple Silicon acceleration by checking the active model process, Ollama logs, and GPU activity during generation.
4. Install and configure the RAG interface.
5. Index a 50-file sample first.
6. Test citation and source highlighting before touching the full corpus.
7. Tune chunk size, top-K, and hybrid search settings against your 10 queries.
8. Run the full index only after the sample phase is stable.
9. Deliver a plain-English handoff document and walkthrough.

For failed embedding steps, I would avoid a single all-or-nothing run. I would use resumable indexing with document hashes, chunk status, and a retry path so completed documents are not embedded again after a failure.

For the full 40,000-file corpus, I would recommend a paid discovery/sanity milestone first because indexing time, disk usage, PDF text quality, and citation highlighting need to be validated on the actual machine.

Suggested milestone:

- Install and verify Ollama/Kotaemon
- Index 50 sample files
- Run your 10 test queries
- Confirm citation behavior
- Provide a fixed quote for full corpus indexing

