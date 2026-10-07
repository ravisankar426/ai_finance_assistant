"""Retrieval evaluation (REQ-RAG-06): recall@5, MRR, off-topic rejection — per method.

    uv run python scripts/eval_retrieval.py            # hybrid vs vector-only vs BM25-only
    uv run python scripts/eval_retrieval.py --sweep    # also sweep the min_cosine gate

Needs OPENAI_API_KEY (query embeddings; ~40 short queries, well under $0.001).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.core.config import get_settings
from src.core.llm import get_embeddings
from src.rag.index import load_or_build
from src.rag.knowledge_base import load_articles
from src.rag.retriever import HybridRetriever
from src.utils.logging import configure_logging

CASES = Path(__file__).resolve().parents[1] / "tests" / "evals" / "retrieval_set.yaml"
K = 5


class _NoEmbeddings:
    """Forces the retriever's BM25-only (degraded) path."""

    def embed_query(self, text: str) -> list[float]:
        raise ConnectionError("disabled for evaluation")


def rank_of_first_hit(articles: list[str], expected: list[str]) -> int | None:
    """1-based rank of the first expected article among *distinct* retrieved articles."""
    seen: list[str] = []
    for a in articles:
        if a not in seen:
            seen.append(a)
    for i, a in enumerate(seen, start=1):
        if a in expected:
            return i
    return None


def evaluate(
    retriever: Any, cases: list[dict[str, Any]], *, vector_only: bool = False
) -> dict[str, float]:
    on_topic = [c for c in cases if c["expected"]]
    off_topic = [c for c in cases if not c["expected"]]
    hits, rr = 0, 0.0
    by_kind: dict[str, list[int]] = {}
    for c in on_topic:
        if vector_only:
            qv = np.asarray(retriever.embeddings.embed_query(c["q"]), dtype=np.float32)
            ids = [retriever.chunks[i].article_id for i, _ in retriever.store.search(qv, K * 4)]
        else:
            ids = [r.article_id for r in retriever.retrieve(c["q"], k=K * 4)]
        rank = rank_of_first_hit(ids, c["expected"])
        ok = rank is not None and rank <= K
        hits += ok
        rr += 1 / rank if rank else 0.0
        by_kind.setdefault(c["kind"], []).append(int(ok))
    rejected = (
        sum(1 for c in off_topic if not retriever.retrieve(c["q"], k=K)) if not vector_only else 0
    )
    out = {"recall@5": hits / len(on_topic), "mrr": rr / len(on_topic)}
    out |= {f"recall@5[{k}]": sum(v) / len(v) for k, v in by_kind.items()}
    if not vector_only:
        out["off_topic_rejected"] = rejected / len(off_topic)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true", help="sweep min_cosine values")
    args = parser.parse_args()
    configure_logging(level="ERROR")  # degraded-mode runs would log a warning per query
    settings, rag = get_settings(), get_settings().rag
    cases = yaml.safe_load(CASES.read_text())
    embeddings = get_embeddings(settings)
    store = load_or_build(
        load_articles(rag.knowledge_base_dir), embeddings, settings.llm.embeddings.model, rag
    )

    def make(emb: Any, min_cosine: float = rag.min_cosine) -> HybridRetriever:
        return HybridRetriever(
            store,
            emb,
            fetch_k=rag.fetch_k,
            rrf_k=rag.rrf_k,
            min_cosine=min_cosine,
            min_keyword_coverage=rag.min_keyword_coverage,
        )

    print(f"{len(cases)} cases, {len(store.chunks)} chunks, min_cosine={rag.min_cosine}\n")
    rows = {
        "hybrid (BM25 + FAISS, RRF)": evaluate(make(embeddings), cases),
        "vector only (FAISS)": evaluate(make(embeddings), cases, vector_only=True),
        "BM25 only (degraded mode)": evaluate(make(_NoEmbeddings()), cases),
    }
    for name, metrics in rows.items():
        print(name)
        for key, value in metrics.items():
            print(f"   {key:<24} {value:.3f}")
    if args.sweep:
        print("\nmin_cosine sweep (hybrid): threshold -> recall@5 / off-topic rejected")
        for t in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
            m = evaluate(make(embeddings, t), cases)
            print(f"   {t:.2f} -> {m['recall@5']:.3f} / {m['off_topic_rejected']:.3f}")


if __name__ == "__main__":
    main()
