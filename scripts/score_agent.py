"""Aggregate agent traces and explicit source-audit labels without LLM grading."""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from src.benchmark import load_questions
from src.retrieval import evidence_hit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {q["id"]: q for q in load_questions(args.split)}
    traces = {t["query_id"]: t for t in json.loads(args.traces.read_text(encoding="utf-8"))}
    reviews = {r["query_id"]: r for r in json.loads(args.reviews.read_text(encoding="utf-8"))}
    if set(cases) != set(traces) or set(cases) != set(reviews):
        raise ValueError("Cases, traces and reviews must contain the same query IDs")
    calls = Counter()
    grades = Counter()
    errors = 0
    tool_use = 0
    citations_total = citations_supported = 0
    answers = answers_cited = 0
    absent_total = absent_abstain = 0
    answerable_total = false_abstain = 0
    reader_evidence = 0
    elapsed = []
    for cid, case in cases.items():
        t, r = traces[cid], reviews[cid]
        if t["status"] == "error":
            errors += 1
            continue
        elapsed.append(t["elapsed_ms"])
        count = sum(e.get("event") == "search" for e in t["events"])
        calls[count] += 1
        tool_use += count > 0
        if t["status"] == "answered":
            answers += 1
            answers_cited += bool(t["citations"])
        support = r.get("citation_support")
        if not isinstance(support, list) or len(support) != len(t["citations"]):
            raise ValueError(f"{cid}: citation_support mismatch")
        if any(type(x) is not bool for x in support):
            raise ValueError(f"{cid}: citation support must be Boolean")
        citations_total += len(support)
        citations_supported += sum(support)
        if case.get("answerable", True):
            answerable_total += 1
            reader_ids = {chunk_id for event in t["events"] if event.get("event") == "reader"
                          for chunk_id in event["candidate_ids"]}
            reader_evidence += any(evidence_hit(chunk, case["gold"]) and chunk["id"] in reader_ids
                                   for event in t["events"] if event.get("event") == "search"
                                   for chunk in event["results"])
            grade = r.get("answer_grade")
            if type(grade) is not int or grade not in (0, 1, 2):
                raise ValueError(f"{cid}: answer_grade must be 0, 1 or 2")
            if t["status"] == "abstain" and grade != 0:
                raise ValueError(f"{cid}: false abstention must be grade 0")
            grades[grade] += 1
            false_abstain += t["status"] == "abstain"
        else:
            absent_total += 1
            correct = r.get("abstention_correct")
            if type(correct) is not bool:
                raise ValueError(f"{cid}: abstention_correct must be Boolean")
            absent_abstain += correct
    summary = {"split": args.split, "cases": len(cases), "provider_errors": errors,
               "successful": len(cases) - errors,
               "tool_use": {"n": tool_use, "N": len(cases) - errors},
               "search_calls": dict(sorted(calls.items())),
               "grounded_answer": {"0": grades[0], "1": grades[1], "2": grades[2],
                                   "N": answerable_total,
                                   "mean_of_2": (grades[1] + 2 * grades[2]) / answerable_total
                                   if answerable_total else None},
               "citation_supported": {"n": citations_supported, "N": citations_total},
               "answers_cited": {"n": answers_cited, "N": answers},
               "correct_abstention": {"n": absent_abstain, "N": absent_total},
               "false_abstention": {"n": false_abstain, "N": answerable_total},
               "reader_context_evidence": {"n": reader_evidence, "N": answerable_total},
               "median_elapsed_ms": statistics.median(elapsed) if elapsed else None,
               "service_cost_usd": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
