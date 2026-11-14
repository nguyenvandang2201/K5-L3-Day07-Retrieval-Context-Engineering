"""Measure retrieval and packed context on an explicit benchmark split."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from .retrieval import ROOT, RetrievalSystem, evidence_hit, pack_context


def load_questions(split: str) -> list[dict]:
    questions = json.loads((ROOT / "data/questions.json").read_text(encoding="utf-8"))
    return [q for q in questions if q["split"] == split]


def run(system: RetrievalSystem, questions: list[dict], method: str,
        top_k: int, budget_chars: int) -> tuple[dict, list[dict]]:
    traces = []
    for q in questions:
        started = time.perf_counter()
        results = system.search(q["question"], method, max(3, top_k), q.get("filter"))
        context, ids = pack_context(results[:top_k], budget_chars)
        elapsed = (time.perf_counter() - started) * 1000
        trace = {"query_id": q["id"], "split": q["split"], "question": q["question"],
                 "method": method, "top_k": top_k, "budget_chars": budget_chars,
                 "filter": q.get("filter"), "results": results, "context": context,
                 "context_ids": ids, "elapsed_ms": round(elapsed, 2)}
        if q.get("answerable", True):
            hits = [evidence_hit(c, q["gold"]) for c in results]
            trace.update({"hit1": bool(hits and hits[0]), "hit3": any(hits[:3]),
                          "context_hit": any(evidence_hit(c, q["gold"]) and c["id"] in ids
                                             for c in results), "gold": q["gold"]})
        traces.append(trace)
    answerable = [t for t in traces if "hit1" in t]
    n = len(answerable)
    summary = {"method": method, "split": questions[0]["split"], "top_k": top_k,
               "budget_chars": budget_chars, "answerable_n": n,
               "unanswerable_n": len(traces) - n,
               "hit1": sum(t["hit1"] for t in answerable),
               "hit3": sum(t["hit3"] for t in answerable),
               "context_hit": sum(t["context_hit"] for t in answerable),
               "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2),
               "service_cost_usd": 0}
    return summary, traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--method", choices=("dense", "bm25", "hybrid"), required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget-chars", type=int, default=2400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k < 1 or args.budget_chars < 1:
        raise ValueError("top-k and budget must be positive")
    questions = load_questions(args.split)
    system = RetrievalSystem(use_dense=args.method != "bm25")
    summary, traces = run(system, questions, args.method, args.top_k, args.budget_chars)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
