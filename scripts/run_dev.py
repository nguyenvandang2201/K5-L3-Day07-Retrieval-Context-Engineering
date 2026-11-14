"""Compare three retrieval methods and two larger-context variants on dev only."""
import json
from datetime import datetime, timezone

from src.benchmark import load_questions, run
from src.retrieval import ROOT, RetrievalSystem

CONFIGS = [
    ("dense_base", "dense", 3, 2400),
    ("bm25_base", "bm25", 3, 2400),
    ("hybrid_base", "hybrid", 3, 2400),
    ("bm25_wide", "bm25", 6, 4000),
    ("hybrid_wide", "hybrid", 6, 4000),
]


def main() -> None:
    questions = load_questions("dev")
    system = RetrievalSystem()
    rows = []
    for name, method, top_k, budget in CONFIGS:
        summary, traces = run(system, questions, method, top_k, budget)
        summary["configuration"] = name
        rows.append(summary)
        folder = ROOT / "results/dev" / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (folder / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(name, summary["hit1"], summary["hit3"], summary["context_hit"], summary["median_ms"])
    # Selection rule is fixed before looking at test: context, top-3, top-1,
    # then the lower median search latency. The baseline remains dense_base.
    chosen = max(rows, key=lambda r: (r["context_hit"], r["hit3"], r["hit1"], -r["median_ms"]))
    selection = {"selected_at_utc": datetime.now(timezone.utc).isoformat(),
                 "rule": "max context_hit, hit3, hit1; min median_ms",
                 "baseline": "dense_base", "chosen": chosen["configuration"],
                 "configurations": rows}
    (ROOT / "results/selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print("chosen", chosen["configuration"])


if __name__ == "__main__":
    main()
