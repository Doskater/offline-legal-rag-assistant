#!/usr/bin/env python3
"""Offline legal RAG demo with resumable indexing and hybrid retrieval."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html.parser
import json
import math
import os
import re
import sqlite3
import textwrap
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB = PROJECT_ROOT / "legal_x_demo.db"
DEFAULT_CORPUS = PROJECT_ROOT / "sample_corpus"
DEFAULT_OUTPUT = PROJECT_ROOT / "sample_output"
DEFAULT_PREFS = PROJECT_ROOT / "preferences.md"
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]+")


class HTMLTextExtractor(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        cleaned = data.strip()
        if cleaned:
            self.parts.append(cleaned)

    def text(self) -> str:
        return "\n".join(self.parts)


@dataclass
class SearchResult:
    chunk_id: int
    source_path: str
    chunk_index: int
    text: str
    bm25: float
    vector: float
    score: float


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        parser = HTMLTextExtractor()
        parser.feed(raw)
        return parser.text()
    return raw


def chunk_text(text: str, chunk_words: int = 120, overlap_words: int = 25) -> list[str]:
    words = normalize_text(text).split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_words])
        if len(chunk.split()) >= 12:
            chunks.append(chunk)
    return chunks


def hash_embedding(text: str, dims: int = 96) -> list[float]:
    vector = [0.0] * dims
    terms = tokenize(text)
    for term in terms:
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * (1.0 + math.log1p(len(term)))
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def ollama_embedding(text: str, model: str, host: str) -> list[float] | None:
    payload = json.dumps({"model": model, "input": text}).encode("utf-8")
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    embeddings = data.get("embeddings") or []
    if not embeddings:
        return None
    return [float(value) for value in embeddings[0]]


def embed_text(text: str, use_ollama: bool, model: str, host: str) -> tuple[list[float], str]:
    if use_ollama:
        vector = ollama_embedding(text, model=model, host=host)
        if vector:
            return vector, f"ollama:{model}"
    return hash_embedding(text), "local-hash"


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            sha256 TEXT NOT NULL,
            status TEXT NOT NULL,
            indexed_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            term_counts TEXT NOT NULL,
            embedding TEXT NOT NULL,
            embedding_backend TEXT NOT NULL,
            status TEXT NOT NULL,
            UNIQUE(document_id, chunk_index),
            FOREIGN KEY(document_id) REFERENCES documents(id)
        )
        """
    )
    return conn


def discover_files(corpus_dir: Path) -> list[Path]:
    allowed = {".md", ".txt", ".html", ".htm"}
    return sorted(path for path in corpus_dir.rglob("*") if path.is_file() and path.suffix.lower() in allowed)


def index_corpus(
    corpus_dir: Path,
    db_path: Path,
    chunk_words: int,
    overlap_words: int,
    use_ollama: bool,
    embedding_model: str,
    ollama_host: str,
) -> dict[str, object]:
    conn = connect_db(db_path)
    stats = {
        "files_seen": 0,
        "files_indexed": 0,
        "files_skipped": 0,
        "chunks_indexed": 0,
        "failed_files": 0,
        "embedding_backend": None,
    }
    for path in discover_files(corpus_dir):
        stats["files_seen"] += 1
        relative = str(path.relative_to(corpus_dir))
        sha = file_hash(path)
        existing = conn.execute(
            "SELECT id, sha256, status FROM documents WHERE path = ?",
            (relative,),
        ).fetchone()
        if existing and existing[1] == sha and existing[2] == "indexed":
            stats["files_skipped"] += 1
            continue
        try:
            text = read_document(path)
            chunks = chunk_text(text, chunk_words=chunk_words, overlap_words=overlap_words)
            with conn:
                if existing:
                    document_id = int(existing[0])
                    conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                    conn.execute(
                        "UPDATE documents SET sha256 = ?, status = ?, indexed_at = ? WHERE id = ?",
                        (sha, "pending", time.time(), document_id),
                    )
                else:
                    cursor = conn.execute(
                        "INSERT INTO documents(path, sha256, status, indexed_at) VALUES (?, ?, ?, ?)",
                        (relative, sha, "pending", time.time()),
                    )
                    document_id = int(cursor.lastrowid)
            for index, chunk in enumerate(chunks):
                terms = Counter(tokenize(chunk))
                vector, backend = embed_text(
                    chunk,
                    use_ollama=use_ollama,
                    model=embedding_model,
                    host=ollama_host,
                )
                stats["embedding_backend"] = backend
                with conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO chunks(
                            document_id, chunk_index, text, token_count,
                            term_counts, embedding, embedding_backend, status
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            document_id,
                            index,
                            chunk,
                            len(terms),
                            json.dumps(terms, sort_keys=True),
                            json.dumps(vector),
                            backend,
                            "embedded",
                        ),
                    )
                stats["chunks_indexed"] += 1
            with conn:
                conn.execute(
                    "UPDATE documents SET status = ?, indexed_at = ? WHERE id = ?",
                    ("indexed", time.time(), document_id),
                )
            stats["files_indexed"] += 1
        except Exception as exc:  # noqa: BLE001 - status capture is intentional for handoff.
            stats["failed_files"] += 1
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO documents(path, sha256, status, indexed_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (relative, sha, f"failed:{type(exc).__name__}", time.time()),
                )
    available = conn.execute(
        """
        SELECT COUNT(*), COALESCE(MAX(embedding_backend), '')
        FROM chunks
        JOIN documents ON documents.id = chunks.document_id
        WHERE chunks.status = 'embedded' AND documents.status = 'indexed'
        """
    ).fetchone()
    stats["chunks_available"] = int(available[0] or 0)
    if not stats["embedding_backend"] and available[1]:
        stats["embedding_backend"] = available[1]
    stats.update(database=str(db_path), corpus=str(corpus_dir))
    conn.close()
    return stats


def load_chunks(conn: sqlite3.Connection) -> list[dict[str, object]]:
    rows = conn.execute(
        """
        SELECT chunks.id, documents.path, chunks.chunk_index, chunks.text,
               chunks.term_counts, chunks.embedding
        FROM chunks
        JOIN documents ON documents.id = chunks.document_id
        WHERE chunks.status = 'embedded' AND documents.status = 'indexed'
        """
    ).fetchall()
    return [
        {
            "id": row[0],
            "path": row[1],
            "chunk_index": row[2],
            "text": row[3],
            "terms": json.loads(row[4]),
            "embedding": json.loads(row[5]),
        }
        for row in rows
    ]


def bm25_scores(query_terms: list[str], chunks: list[dict[str, object]]) -> dict[int, float]:
    if not chunks:
        return {}
    doc_count = len(chunks)
    avg_len = sum(sum(terms.values()) for terms in (chunk["terms"] for chunk in chunks)) / doc_count
    document_frequency: Counter[str] = Counter()
    for chunk in chunks:
        terms = chunk["terms"]
        for term in set(terms):
            document_frequency[term] += 1
    scores: dict[int, float] = {}
    k1 = 1.5
    b = 0.75
    for chunk in chunks:
        terms = chunk["terms"]
        length = sum(terms.values()) or 1
        score = 0.0
        for term in query_terms:
            frequency = terms.get(term, 0)
            if not frequency:
                continue
            idf = math.log(1 + (doc_count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            numerator = frequency * (k1 + 1)
            denominator = frequency + k1 * (1 - b + b * length / avg_len)
            score += idf * numerator / denominator
        scores[int(chunk["id"])] = score
    return scores


def normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    max_score = max(scores.values()) or 1.0
    return {key: value / max_score for key, value in scores.items()}


def search(
    query: str,
    db_path: Path,
    top_k: int,
    vector_weight: float,
    use_ollama: bool,
    embedding_model: str,
    ollama_host: str,
) -> list[SearchResult]:
    conn = connect_db(db_path)
    chunks = load_chunks(conn)
    conn.close()
    query_terms = tokenize(query)
    query_vector, _ = embed_text(query, use_ollama=use_ollama, model=embedding_model, host=ollama_host)
    bm25 = normalize_scores(bm25_scores(query_terms, chunks))
    vector = normalize_scores(
        {
            int(chunk["id"]): max(0.0, cosine(query_vector, chunk["embedding"]))
            for chunk in chunks
        }
    )
    results: list[SearchResult] = []
    for chunk in chunks:
        chunk_id = int(chunk["id"])
        bm25_score = bm25.get(chunk_id, 0.0)
        vector_score = vector.get(chunk_id, 0.0)
        score = (1 - vector_weight) * bm25_score + vector_weight * vector_score
        if score <= 0:
            continue
        results.append(
            SearchResult(
                chunk_id=chunk_id,
                source_path=str(chunk["path"]),
                chunk_index=int(chunk["chunk_index"]),
                text=str(chunk["text"]),
                bm25=bm25_score,
                vector=vector_score,
                score=score,
            )
        )
    return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]


def load_preferences(path: Path) -> str:
    if not path.exists():
        return ""
    return normalize_text(path.read_text(encoding="utf-8", errors="ignore"))


def answer_query(query: str, results: list[SearchResult], preferences: str) -> str:
    if not results:
        return f"# Answer\n\nNo indexed source passage matched the query: {query}\n"
    style_note = "Short answer first, then numbered source notes."
    if "plain English" in preferences:
        style_note += " Plain-English drafting style requested."
    lead = extract_short_answer(query, results[0].text)
    answer_lines = [
        "# Answer",
        "",
        f"**Query:** {query}",
        "",
        f"**Short answer:** {lead}",
        "",
        f"Style applied: {style_note}",
        "",
        "## Footnotes",
        "",
    ]
    for index, result in enumerate(results, start=1):
        passage = textwrap.shorten(result.text, width=420, placeholder="...")
        answer_lines.append(
            f"[{index}] `{result.source_path}` chunk {result.chunk_index}: {passage}"
        )
    answer_lines.extend(["", "## Retrieval Scores", ""])
    for index, result in enumerate(results, start=1):
        answer_lines.append(
            f"- [{index}] hybrid={result.score:.3f}, bm25={result.bm25:.3f}, vector={result.vector:.3f}"
        )
    return "\n".join(answer_lines) + "\n"


def extract_short_answer(query: str, passage: str) -> str:
    query_terms = set(tokenize(query))
    passage = re.sub(r"#+\s*", "", passage)
    sentences = re.split(r"(?<=[.!?])\s+", normalize_text(passage))
    ranked: list[tuple[int, str]] = []
    for sentence in sentences:
        terms = set(tokenize(sentence))
        overlap = len(query_terms & terms)
        if overlap:
            ranked.append((overlap, sentence))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = [sentence for _, sentence in ranked[:2]]
    if not selected:
        selected = sentences[:1]
    return " ".join(selected)


def evaluate_queries(
    queries_path: Path,
    db_path: Path,
    output_dir: Path,
    top_k: int,
    vector_weight: float,
    use_ollama: bool,
    embedding_model: str,
    ollama_host: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with queries_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            query = row["query"]
            expected_terms = [term.strip().lower() for term in row["expected_terms"].split("|")]
            results = search(
                query,
                db_path=db_path,
                top_k=top_k,
                vector_weight=vector_weight,
                use_ollama=use_ollama,
                embedding_model=embedding_model,
                ollama_host=ollama_host,
            )
            combined = " ".join(result.text.lower() for result in results)
            matched = sum(1 for term in expected_terms if term in combined)
            passed = matched >= max(1, math.ceil(len(expected_terms) * 0.66))
            rows.append(
                {
                    "query": query,
                    "top_source": results[0].source_path if results else "",
                    "top_score": f"{results[0].score:.3f}" if results else "0.000",
                    "expected_terms": "|".join(expected_terms),
                    "matched_terms": matched,
                    "passed": "yes" if passed else "no",
                }
            )
    csv_path = output_dir / "ranked_queries.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    passed_count = sum(1 for row in rows if row["passed"] == "yes")
    report = {
        "queries": len(rows),
        "passed": passed_count,
        "pass_rate": round(passed_count / len(rows), 3) if rows else 0.0,
        "top_k": top_k,
        "vector_weight": vector_weight,
    }
    daily_report = output_dir / "daily_report.md"
    daily_report.write_text(render_daily_report(rows, report), encoding="utf-8")
    return report


def render_daily_report(rows: list[dict[str, object]], report: dict[str, object]) -> str:
    lines = [
        "# Offline Legal RAG Evaluation Report",
        "",
        f"- Queries tested: {report['queries']}",
        f"- Passed: {report['passed']}",
        f"- Pass rate: {report['pass_rate']}",
        f"- Top-K: {report['top_k']}",
        f"- Vector weight: {report['vector_weight']}",
        "",
        "## Query Results",
        "",
        "| Query | Top Source | Score | Passed |",
        "| --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['query']} | `{row['top_source']}` | {row['top_score']} | {row['passed']} |"
        )
    return "\n".join(lines) + "\n"


def write_index_stats(stats: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def command_index(args: argparse.Namespace) -> None:
    stats = index_corpus(
        corpus_dir=Path(args.corpus),
        db_path=Path(args.db),
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        use_ollama=args.use_ollama,
        embedding_model=args.embedding_model,
        ollama_host=args.ollama_host,
    )
    write_index_stats(stats, Path(args.output))
    print(json.dumps(stats, indent=2))


def command_ask(args: argparse.Namespace) -> None:
    results = search(
        args.query,
        db_path=Path(args.db),
        top_k=args.top_k,
        vector_weight=args.vector_weight,
        use_ollama=args.use_ollama,
        embedding_model=args.embedding_model,
        ollama_host=args.ollama_host,
    )
    answer = answer_query(args.query, results, load_preferences(Path(args.preferences)))
    output_dir = Path(args.output)
    output = output_dir / "sample_answer.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(answer, encoding="utf-8")
    print(answer)


def command_eval(args: argparse.Namespace) -> None:
    report = evaluate_queries(
        queries_path=Path(args.queries),
        db_path=Path(args.db),
        output_dir=Path(args.output),
        top_k=args.top_k,
        vector_weight=args.vector_weight,
        use_ollama=args.use_ollama,
        embedding_model=args.embedding_model,
        ollama_host=args.ollama_host,
    )
    print(json.dumps(report, indent=2))


def command_demo(args: argparse.Namespace) -> None:
    stats = index_corpus(
        corpus_dir=Path(args.corpus),
        db_path=Path(args.db),
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        use_ollama=args.use_ollama,
        embedding_model=args.embedding_model,
        ollama_host=args.ollama_host,
    )
    output_dir = Path(args.output)
    write_index_stats(stats, output_dir)
    evaluate_queries(
        queries_path=Path(args.queries),
        db_path=Path(args.db),
        output_dir=output_dir,
        top_k=args.top_k,
        vector_weight=args.vector_weight,
        use_ollama=args.use_ollama,
        embedding_model=args.embedding_model,
        ollama_host=args.ollama_host,
    )
    results = search(
        args.query,
        db_path=Path(args.db),
        top_k=args.top_k,
        vector_weight=args.vector_weight,
        use_ollama=args.use_ollama,
        embedding_model=args.embedding_model,
        ollama_host=args.ollama_host,
    )
    answer = answer_query(args.query, results, load_preferences(Path(args.preferences)))
    (output_dir / "sample_answer.md").write_text(answer, encoding="utf-8")
    print(f"Demo complete. Outputs written to {output_dir}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--chunk-words", type=int, default=120)
    parser.add_argument("--overlap-words", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--vector-weight", type=float, default=0.35)
    parser.add_argument("--use-ollama", action="store_true")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    index_parser = subparsers.add_parser("index", help="Index or resume indexing the local corpus")
    add_common_args(index_parser)
    index_parser.set_defaults(func=command_index)

    ask_parser = subparsers.add_parser("ask", help="Ask a source-grounded question")
    add_common_args(ask_parser)
    ask_parser.add_argument("query")
    ask_parser.add_argument("--preferences", default=str(DEFAULT_PREFS))
    ask_parser.set_defaults(func=command_ask)

    eval_parser = subparsers.add_parser("eval", help="Run the 10-query retrieval evaluation")
    add_common_args(eval_parser)
    eval_parser.add_argument("--queries", default=str(PROJECT_ROOT / "sample_queries.csv"))
    eval_parser.set_defaults(func=command_eval)

    demo_parser = subparsers.add_parser("demo", help="Run index, evaluation, and sample answer")
    add_common_args(demo_parser)
    demo_parser.add_argument("--queries", default=str(PROJECT_ROOT / "sample_queries.csv"))
    demo_parser.add_argument("--preferences", default=str(DEFAULT_PREFS))
    demo_parser.add_argument(
        "--query",
        default="What does the handbook say about response times for data requests?",
    )
    demo_parser.set_defaults(func=command_demo)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
