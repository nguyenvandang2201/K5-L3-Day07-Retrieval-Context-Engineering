"""Run the frozen dev selection and baseline on the held-out test split."""
import json

from src.benchmark import load_questions, run
from src.retrieval import ROOT, RetrievalSystem

CONFIGS = {"dense_base": ("dense", 3, 2400), "bm25_base": ("bm25", 3, 2400),
           "hybrid_base": ("hybrid", 3, 2400), "bm25_wide": ("bm25", 6, 4000),
           "hybrid_wide": ("hybrid", 6, 4000)}


def main() -> None:
    selection = json.loads((ROOT / "results/selection.json").read_text(encoding="utf-8"))
    questions = load_questions("test")
    system = RetrievalSystem()
    for name in dict.fromkeys((selection["baseline"], selection["chosen"])):
        method, top_k, budget = CONFIGS[name]
        summary, traces = run(system, questions, method, top_k, budget)
        summary["configuration"] = name
        folder = ROOT / "results/test" / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (folder / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(name, summary["hit1"], summary["hit3"], summary["context_hit"], summary["median_ms"])


if __name__ == "__main__":
    main()
