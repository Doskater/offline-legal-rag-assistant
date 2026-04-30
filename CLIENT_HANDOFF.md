# Client Handoff: Offline Legal RAG Assistant

## Start The Demo

```bash
cd /Users/admin/Documents/ai-freelance-portfolio/offline-legal-rag-assistant
python3 rag_assistant.py demo
```

## Add New Files

Place `.md`, `.txt`, `.html`, or `.htm` files in:

```text
sample_corpus/
```

Then re-run:

```bash
python3 rag_assistant.py index
```

The indexer stores a SHA-256 hash for every file. Unchanged files are skipped. Changed files are re-indexed.

## Re-Index From Scratch

Delete the demo database:

```bash
rm legal_x_demo.db
```

Then run:

```bash
python3 rag_assistant.py demo
```

## Ask A Question

```bash
python3 rag_assistant.py ask "What is required before using AI-generated drafts externally?"
```

The answer is saved to:

```text
sample_output/sample_answer.md
```

## Run Evaluation

```bash
python3 rag_assistant.py eval
```

Review:

```text
sample_output/daily_report.md
sample_output/ranked_queries.csv
```

## Use Ollama Embeddings

```bash
ollama pull nomic-embed-text
python3 rag_assistant.py demo --use-ollama --embedding-model nomic-embed-text
```

If Ollama is unavailable, the script automatically falls back to local deterministic embeddings.

## Verify Apple Silicon Acceleration In A Real Setup

For Ollama on Apple Silicon:

1. Confirm the Mac is Apple Silicon:
   ```bash
   uname -m
   ```
2. Run a model:
   ```bash
   ollama run qwen2.5:14b "Reply with one sentence."
   ```
3. Check loaded models:
   ```bash
   ollama ps
   ```
4. Review Ollama logs if performance looks wrong:
   ```bash
   tail -n 100 ~/.ollama/logs/server.log
   ```
5. Watch GPU usage in Activity Monitor during generation.

Ollama supports Apple GPU acceleration through Metal on Apple Silicon; no CUDA-style setup is required.

## Failed Embedding Recovery

The SQLite index tracks document status and chunk status. The intended recovery workflow is:

1. Keep completed file/chunk records.
2. Mark failed files as `failed:<error>`.
3. Fix the input file, dependency, or model issue.
4. Re-run `python3 rag_assistant.py index`.
5. Unchanged indexed files are skipped; changed or failed files can be retried.

For a larger production system, this pattern should be expanded with:

- per-chunk retry counters
- failed chunk queue
- batch IDs
- index progress dashboard
- periodic backups of the vector store

## Common Troubleshooting

### No Results

Run indexing first:

```bash
python3 rag_assistant.py index
```

### Ollama Embeddings Are Slow

Use the local fallback for the demo or switch to a smaller embedding model.

### Source Passage Looks Too Broad

Lower chunk size:

```bash
python3 rag_assistant.py demo --chunk-words 80 --overlap-words 20
```

### Retrieval Is Too Keyword-Heavy

Increase vector weight:

```bash
python3 rag_assistant.py eval --vector-weight 0.5
```

### Retrieval Is Too Semantic And Misses Exact Terms

Lower vector weight:

```bash
python3 rag_assistant.py eval --vector-weight 0.2
```

