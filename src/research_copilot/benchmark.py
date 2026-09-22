"""Labelled retrieval ablations and answer checks with honest denominators."""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
import time

from .config import Settings
from .service import Copilot

CASES = Path(__file__).with_name("benchmark_cases.json")
VARIANTS = {"bm25": ("bm25", False), "dense": ("dense", False),
            "hybrid": ("hybrid", False), "hybrid_rerank": ("hybrid", True)}


def ranking_metrics(documents, relevant, k=3):
    """Document-level binary relevance: repeated chunks never count twice."""
    ranked = list(dict.fromkeys(documents))[:k]
    gold = set(relevant)
    if not gold:
        return None
    hits = [int(name in gold) for name in ranked]
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return {"recall_at_k": sum(hits) / len(gold),
            "mrr_at_k": next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0),
            "ndcg_at_k": sum(hit / math.log2(i + 2) for i, hit in enumerate(hits)) / ideal}


def validate_dataset(dataset):
    docs, cases = dataset["documents"], dataset["queries"]
    names = {doc["name"] for doc in docs}
    if len(names) != len(docs) or not 1 <= len(docs) <= 50:
        raise ValueError("Expected 1..50 uniquely named documents")
    ids = {case["id"] for case in cases}
    if len(ids) != len(cases) or not 1 <= len(cases) <= 50:
        raise ValueError("Expected 1..50 uniquely identified queries")
    for case in cases:
        if not set(case["relevant"]) <= names:
            raise ValueError("Relevance labels reference an unknown document")
        if bool(case["relevant"]) != bool(case["answer_terms"]):
            raise ValueError("Answerable queries require answer terms; unanswerable queries must have none")


def rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def run_benchmark(settings=None, dataset_path=CASES, *, variants=None, factory=Copilot):
    raw = Path(dataset_path).read_bytes()
    dataset = json.loads(raw)
    validate_dataset(dataset)
    settings = settings or Settings()
    variants = tuple(variants or VARIANTS)
    if not set(variants) <= VARIANTS.keys():
        raise ValueError("Unknown benchmark variant")
    report = {"data_source": "synthetic_labelled_benchmark", "dataset_version": dataset["version"],
              "dataset_sha256": hashlib.sha256(raw).hexdigest(), "mode": settings.mode,
              "model": settings.model if settings.mode == "qwen" else None,
              "reranker": settings.reranker, "k": 3, "variants": {}, "answers": [],
              "limitations": ["Small authored dataset; not independent business evaluation or model ranking.",
                              "Answer term coverage is not semantic correctness or factual entailment.",
                              "Provider usage covers answer generation only; not total cost.",
                              "Failed calls remain in denominators and are reported explicitly."]}
    with tempfile.TemporaryDirectory(prefix="copilot-benchmark-") as directory:
        app = factory(replace(settings, db_path=str(Path(directory) / "benchmark.db")))
        try:
            report["embedding_fingerprint"] = app.embedder.fingerprint
            for doc in dataset["documents"]:
                app.ingest(doc["name"], doc["text"].encode())
            for variant in variants:
                rows = []
                strategy, rerank = VARIANTS[variant]
                for case in dataset["queries"]:
                    begun = time.perf_counter()
                    row = {"id": case["id"], "relevant": case["relevant"]}
                    try:
                        hits = app.retriever.search(case["question"], 12, strategy=strategy, rerank=rerank)
                        row["documents"] = list(dict.fromkeys(hit["document_name"] for hit in hits))[:3]
                    except Exception as exc:
                        row.update(documents=[], error_type=type(exc).__name__)
                    row["metrics"] = ranking_metrics(row["documents"], case["relevant"])
                    row["latency_ms"] = round((time.perf_counter() - begun) * 1000, 3)
                    rows.append(row)
                measured = [row["metrics"] for row in rows if row["metrics"] is not None]
                report["variants"][variant] = {
                    "rows": rows, "answerable_count": len(measured),
                    "errors": sum("error_type" in row for row in rows),
                    "mean": {key: statistics.mean(row[key] for row in measured) if measured else None
                             for key in ("recall_at_k", "mrr_at_k", "ndcg_at_k")},
                    "median_latency_ms": statistics.median(row["latency_ms"] for row in rows)}
            for case in dataset["queries"]:
                row = {"id": case["id"], "answerable": bool(case["relevant"])}
                begun = time.perf_counter()
                try:
                    result = app.chat(case["question"])
                    row.update(refused=result["refused"], answer=result["answer"],
                               cited_documents=[source["document_name"] for source in result["citations"]],
                               all_labelled_terms_present=not result["refused"] and bool(case["answer_terms"]) and
                               all(term.casefold() in result["answer"].casefold() for term in case["answer_terms"]),
                               source_integrity=result["trace"]["citation_validation"],
                               provider_usage=result["trace"]["usage"])
                except Exception as exc:
                    row["error_type"] = type(exc).__name__
                row["latency_ms"] = round((time.perf_counter() - begun) * 1000, 3)
                report["answers"].append(row)
            answerable = [row for row in report["answers"] if row["answerable"]]
            unknown = [row for row in report["answers"] if not row["answerable"]]
            answered = [row for row in report["answers"] if row.get("refused") is False]
            report["answer_metrics"] = {
                "labelled_term_coverage": rate(sum(row.get("all_labelled_terms_present", False) for row in answerable), len(answerable)),
                "answerable_non_refusal": rate(sum(row.get("refused") is False for row in answerable), len(answerable)),
                "unanswerable_refusal": rate(sum(row.get("refused") is True for row in unknown), len(unknown)),
                "answered_source_integrity": rate(sum(row.get("source_integrity") == "source_id_and_verbatim_quote_passed" for row in answered), len(answered)),
                "errors": sum("error_type" in row for row in report["answers"])}
        finally:
            app.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["offline", "qwen"], default="offline")
    parser.add_argument("--allow-provider-calls", action="store_true")
    parser.add_argument("--dataset", type=Path, default=CASES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "qwen" and not args.allow_provider_calls:
        parser.error("Qwen sends this dataset to the configured service and may incur API charges; pass --allow-provider-calls explicitly")
    settings = Settings() if args.mode == "offline" else Settings.from_env()
    if settings.mode != args.mode:
        parser.error("COPILOT_MODE must equal the requested --mode")
    report = run_benchmark(settings, args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"mode": report["mode"], "answer_metrics": report["answer_metrics"],
                      "retrieval": {name: value["mean"] for name, value in report["variants"].items()}}, ensure_ascii=True))
    # Low quality scores are findings; infrastructure failures must fail CI.
    return int(bool(report["answer_metrics"]["errors"] or any(value["errors"] for value in report["variants"].values())))


if __name__ == "__main__":
    raise SystemExit(main())
