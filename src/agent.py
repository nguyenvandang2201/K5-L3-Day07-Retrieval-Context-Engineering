"""Local model driven retrieval agent with auditable tool and context traces."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from .benchmark import load_questions
from .retrieval import ROOT, RetrievalSystem, pack_context, tokenize
from .qa import ExtractiveReader

STOPWORDS = {"what", "which", "when", "where", "why", "how", "does", "do", "is", "are",
             "a", "an", "the", "in", "on", "for", "of", "to", "with", "from", "by",
             "can", "it", "be", "python", "using", "used", "into"}


def stem(term: str) -> str:
    return term[:-1] if term.endswith("s") and len(term) > 4 else term


def relevant_source(question: str, source: str) -> bool:
    terms = {stem(t) for t in tokenize(question) if t not in STOPWORDS and len(t) > 2}
    present = {stem(t) for t in tokenize(source)}
    return len(terms & present) >= min(2, len(terms))


class LocalModel:
    def __init__(self, name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(min(8, torch.get_num_threads()))
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name)
        self.model.eval()

    def ask(self, system: str, user: str, max_new_tokens: int = 160) -> str:
        import torch
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=True,
                                                    add_generation_prompt=True,
                                                    return_tensors="pt")
        input_ids = inputs["input_ids"] if hasattr(inputs, "keys") else inputs
        with torch.inference_mode():
            output = self.model.generate(input_ids, max_new_tokens=max_new_tokens,
                                         do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()


def parse_object(text: str) -> dict | None:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", text, re.S)
        if match:
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                pass
    return None


def run_one(question: str, system: RetrievalSystem, model: LocalModel, reader: ExtractiveReader, *,
            method: str, top_k: int, budget_chars: int, filters: dict | None = None,
            max_searches: int = 2) -> dict:
    started = time.perf_counter()
    events = []
    context, ids = "", []
    previous_queries: set[str] = set()
    try:
        for attempt in range(max_searches):
            plan_raw = model.ask(
                "You are a Python 3.12 Tutorial research agent. Choose a concise search query "
                "from the user's question. Use only terms the user provided or faithful synonyms. "
                "Return one JSON object only: {\"action\":\"search\",\"query\":\"...\"}.",
                f"Question: {question}\nSearch attempt: {attempt + 1}\n"
                f"Previous context IDs: {ids}", 90)
            plan = parse_object(plan_raw)
            query = plan.get("query") if plan and plan.get("action") == "search" else None
            if not isinstance(query, str) or not query.strip():
                query = question
                events.append({"event": "plan_fallback", "raw": plan_raw})
            else:
                query = query.strip()
            if query.casefold() in previous_queries:
                events.append({"event": "duplicate_query_stop", "query": query})
                break
            previous_queries.add(query.casefold())
            results = system.search(query, method, top_k, filters)
            # Retain candidate order across searches; deduplicate in pack_context.
            if attempt == 0:
                candidates = results
            else:
                candidates = [*results, *candidates]
            context, ids = pack_context(candidates, budget_chars)
            events.append({"event": "search", "method": method, "query": query,
                           "model_action": plan_raw,
                           "filter": filters, "top_k": top_k, "results": results,
                           "context": context, "context_ids": ids})
            if not context:
                break
            readable = [next(c for c in candidates if c["id"] == cid) for cid in ids]
            readable = [c for c in readable if relevant_source(question, c["text"])]
            proposal = reader.read(question, readable, threshold=-3.0) if readable else None
            events.append({"event": "reader", "model": reader.name,
                           "candidate_ids": [c["id"] for c in readable], "proposal": proposal})
            if proposal:
                answer = " ".join(proposal["answer"].split())
                cid = proposal["chunk_id"]
                events.append({"event": "answer", "evidence_quote": proposal["answer"]})
                return {"status": "answered", "answer": f"{answer} [{cid}]",
                        "citations": [cid], "events": events,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}
        events.append({"event": "abstain"})
        return {"status": "abstain", "answer": "Không đủ bằng chứng để trả lời.",
                "citations": [], "events": events,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}
    except Exception as exc:
        events.append({"event": "error", "type": type(exc).__name__, "message": str(exc)})
        return {"status": "error", "answer": "", "citations": [], "events": events,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--configuration", choices=("dense_base", "bm25_base", "hybrid_base",
                                                   "bm25_wide", "hybrid_wide"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configs = {"dense_base": ("dense", 3, 2400), "bm25_base": ("bm25", 3, 2400),
               "hybrid_base": ("hybrid", 3, 2400), "bm25_wide": ("bm25", 6, 4000),
               "hybrid_wide": ("hybrid", 6, 4000)}
    method, top_k, budget = configs[args.configuration]
    questions = load_questions(args.split)
    system = RetrievalSystem(use_dense=method != "bm25")
    model = LocalModel()
    reader = ExtractiveReader()
    traces = []
    for q in questions:
        trace = {"query_id": q["id"], "split": args.split, "question": q["question"],
                 "configuration": args.configuration, "model": model.name,
                 "reader_model": reader.name,
                 **run_one(q["question"], system, model, reader, method=method, top_k=top_k,
                           budget_chars=budget, filters=q.get("filter"))}
        traces.append(trace)
        print(q["id"], trace["status"], trace["answer"][:100], flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
