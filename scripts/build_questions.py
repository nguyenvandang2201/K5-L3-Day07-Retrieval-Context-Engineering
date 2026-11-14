"""Build the Day 7 benchmark from the licensed Day 6 snapshot.

The held-out test questions use source units absent from the Day 6 benchmark.
This script validates evidence against the full source and the paragraph index.
"""
import json
from pathlib import Path

from src.chunking import by_paragraph
from src.models import DocumentPage

ROOT = Path(__file__).resolve().parents[1]

DEV_QUESTIONS = [
    "Which character begins a Python comment?",
    "Which operator performs floor division?",
    "How can a string keep backslashes from being interpreted as escape characters?",
    "Are Python lists mutable?",
    "Is the endpoint of range() included in the generated sequence?",
    "When does a loop's else clause execute?",
    "What does the pass statement do?",
    "Which keyword introduces a function definition?",
    "What do list methods that only modify the list in place return?",
    "Does a list used as a stack follow last-in, first-out order?",
    "Can a set contain duplicate elements?",
    "Must dictionary keys be unique within one dictionary?",
    "What is a Python module?",
    "When are initialization statements in a module executed during import?",
    "Does reading a .pyc file make the program run faster?",
    "How do packages structure Python's module namespace?",
    "How are expressions inserted into an f-string?",
    "Why is the with keyword recommended when working with files?",
]

TEST_QUESTIONS = [
    ("t01", "What happens when a string index is too large?", "IndexError: string index out of range",
     "introduction", 5, "Attempting to use an index that is too large will result in an error", "An IndexError is raised."),
    ("t02", "What does a Python for statement iterate over?", "items of any sequence",
     "controlflow", 1, "iterates over the items of any sequence", "The items of any sequence, in order."),
    ("t03", "Which function returns both an index and value when looping through a sequence?", "enumerate()",
     "datastructures", 8, "using the enumerate() function", "enumerate()."),
    ("t04", "What name does __name__ have when a module is run as a script?", "__main__",
     "modules", 3, 'with the __name__ set to "__main__"', "__main__."),
    ("t05", "Which class should user-defined exceptions typically derive from?", "Exception",
     "errors", 6, "Exceptions should typically be derived from the Exception class", "Exception, directly or indirectly."),
]

ABSENT = [
    ("d_abs01", "dev", "What exact memory limit does the tutorial set for every Python list?",
     "The corpus gives no universal memory limit for lists."),
    ("d_abs02", "dev", "What guaranteed speedup does Python 3.12 give every program?",
     "No universal speedup guarantee appears in the six chapters."),
    ("t_abs01", "test", "What is the warranty period for the Python interpreter?",
     "The corpus has no warranty period for the interpreter."),
    ("t_abs02", "test", "Which company acquired Python in 2025?",
     "The corpus does not state such an acquisition."),
]


def main() -> None:
    original = json.loads((ROOT / "data/day6/questions_original.json").read_text(encoding="utf-8"))
    pages = [DocumentPage(**json.loads(line)) for line in
             (ROOT / "data/day6/corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    by_page = {(p.document_id, p.page): p for p in pages}
    old_pages = {(q["gold_source_id"], q["gold_page"]) for q in original}
    questions = []
    for q, english in zip((x for x in original if x["split"] == "dev"), DEV_QUESTIONS, strict=True):
        questions.append({"id": q["id"], "split": "dev", "question": english,
                          "expected_answer": q["expected_answer"],
                          "gold": [{"source_id": q["gold_source_id"], "page": q["gold_page"],
                                    "evidence": q["evidence_quote"]}],
                          "filter": q.get("filter")})
    for qid, question, keyword, source_id, page, evidence, answer in TEST_QUESTIONS:
        assert (source_id, page) not in old_pages, (source_id, page)
        questions.append({"id": qid, "split": "test", "question": question,
                          "expected_answer": answer,
                          "gold": [{"source_id": source_id, "page": page, "evidence": evidence}],
                          "filter": None})
    for qid, split, question, reason in ABSENT:
        questions.append({"id": qid, "split": split, "question": question,
                          "answerable": False, "expected_behavior": "search_and_abstain",
                          "absence_reason": reason, "filter": None})
    for q in questions:
        for gold in q.get("gold", []):
            source = by_page[(gold["source_id"], gold["page"])]
            normalized = lambda text: " ".join(text.split())
            assert normalized(gold["evidence"]) in normalized(source.text), q["id"]
            assert any(normalized(gold["evidence"]) in normalized(c.text)
                       for c in by_paragraph(source, 650)), q["id"]
    target = ROOT / "data/questions.json"
    target.write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(questions)} questions; dev={sum(q['split']=='dev' for q in questions)}, "
          f"test={sum(q['split']=='test' for q in questions)}")


if __name__ == "__main__":
    main()
