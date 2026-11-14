"""Extractive SQuAD2 reader; every proposed answer is a span of a cited chunk."""
from __future__ import annotations

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

MODEL_NAME = "deepset/roberta-base-squad2"


class ExtractiveReader:
    def __init__(self, name: str = MODEL_NAME):
        torch.set_num_threads(min(8, torch.get_num_threads()))
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(name)
        self.model.eval()

    def read(self, question: str, chunks: list[dict], threshold: float = 2.0) -> dict | None:
        best = None
        with torch.inference_mode():
            for chunk in chunks:
                encoded = self.tokenizer(question, chunk["text"], truncation="only_second",
                                         max_length=512, return_offsets_mapping=True,
                                         return_tensors="pt")
                offsets = encoded.pop("offset_mapping")[0].tolist()
                sequence_ids = encoded.sequence_ids(0)
                output = self.model(**encoded)
                starts = output.start_logits[0]
                ends = output.end_logits[0]
                null = (starts[0] + ends[0]).item()
                span_best = None
                # The chunk is short; exhaustive span scoring avoids an arbitrary top-N cutoff.
                for i, part in enumerate(sequence_ids):
                    if part != 1:
                        continue
                    for j in range(i, min(i + 30, len(sequence_ids))):
                        if sequence_ids[j] != 1:
                            break
                        score = (starts[i] + ends[j]).item() - null
                        if span_best is None or score > span_best["margin"]:
                            span_best = {"margin": score, "start": offsets[i][0],
                                         "end": offsets[j][1], "chunk_id": chunk["id"]}
                if span_best and (best is None or span_best["margin"] > best["margin"]):
                    span_best["answer"] = chunk["text"][span_best["start"]:span_best["end"]]
                    best = span_best
        return best if best and best["margin"] >= threshold and best["answer"].strip() else None
