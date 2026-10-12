"""
scripts/explore_datasets.py
----------------------------
Exploration-only script: downloads RAGTruth and HaluEval, then prints
schema, sample records, and label distributions.  No data is cleaned,
transformed, or loaded into project models.

Run from the raguard/ directory:
    python scripts/explore_datasets.py
"""

import json
import textwrap
from collections import Counter
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
RAGTRUTH_DIR = RAW_DIR / "ragtruth"
HALUEVAL_DIR = RAW_DIR / "halueval"
RAGTRUTH_DIR.mkdir(parents=True, exist_ok=True)
HALUEVAL_DIR.mkdir(parents=True, exist_ok=True)

_SEP = "=" * 80


def _trunc(value, limit=500):
    s = str(value)
    return s if len(s) <= limit else s[:limit] + " ..."


def _print_section(title):
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


def _print_record(rec, idx):
    print(f"\n--- Record {idx} ---")
    if isinstance(rec, dict):
        for k, v in rec.items():
            print(f"  {k}: {_trunc(v)}")
    else:
        print(_trunc(rec))


# ===========================================================================
# RAGTruth
# ===========================================================================

RAGTRUTH_SOURCE_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl"
RAGTRUTH_RESPONSE_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl"


def download_ragtruth():
    import urllib.request

    files = {
        "source_info.jsonl": RAGTRUTH_SOURCE_URL,
        "response.jsonl": RAGTRUTH_RESPONSE_URL,
    }
    for fname, url in files.items():
        dest = RAGTRUTH_DIR / fname
        if dest.exists():
            print(f"  [cache] {dest.name} already present, skipping download.")
        else:
            print(f"  Downloading {url} ...")
            urllib.request.urlretrieve(url, dest)
            print(f"  Saved -> {dest}")
    return files


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def python_type(value):
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, list):
        return f"list[{type(value[0]).__name__ if value else '?'}]"
    return type(value).__name__


def explore_ragtruth():
    _print_section("DATASET: RAGTruth")
    print(f"\nSource repo : https://github.com/ParticleMedia/RAGTruth")
    print(f"Files       : source_info.jsonl, response.jsonl")
    print(f"Download URLs:")
    print(f"  {RAGTRUTH_SOURCE_URL}")
    print(f"  {RAGTRUTH_RESPONSE_URL}")

    download_ragtruth()

    # ---- source_info.jsonl ------------------------------------------------
    _print_section("RAGTruth | source_info.jsonl")
    sources = load_jsonl(RAGTRUTH_DIR / "source_info.jsonl")
    print(f"\nNumber of records: {len(sources)}")

    # Field types from first record
    first_src = sources[0]
    print("\nField names and Python types:")
    for k, v in first_src.items():
        print(f"  {k!r:30s}  {python_type(v)}")

    print("\nFirst 3 records:")
    for i, rec in enumerate(sources[:3]):
        _print_record(rec, i + 1)

    # Task types
    task_counter = Counter(r.get("task_type") for r in sources)
    print(f"\nTask type distribution:")
    for task, cnt in sorted(task_counter.items()):
        print(f"  {task}: {cnt}")

    # Source (dataset provenance)
    src_counter = Counter(r.get("source") for r in sources)
    print(f"\n'source' field distribution (provenance dataset):")
    for s, cnt in sorted(src_counter.items()):
        print(f"  {s}: {cnt}")

    # ---- response.jsonl ---------------------------------------------------
    _print_section("RAGTruth | response.jsonl")
    responses = load_jsonl(RAGTRUTH_DIR / "response.jsonl")
    print(f"\nNumber of records: {len(responses)}")

    first_resp = responses[0]
    print("\nField names and Python types:")
    for k, v in first_resp.items():
        print(f"  {k!r:30s}  {python_type(v)}")

    print("\nFirst 3 records:")
    for i, rec in enumerate(responses[:3]):
        _print_record(rec, i + 1)

    # Label distribution: hallucinated vs not
    hallucinated = sum(1 for r in responses if r.get("labels"))
    not_hallucinated = len(responses) - hallucinated
    print(f"\nResponse-level hallucination distribution (labels list non-empty):")
    print(f"  Hallucinated (labels != []):    {hallucinated}")
    print(f"  Not hallucinated (labels == []): {not_hallucinated}")

    # label_type distribution across all spans
    label_types = Counter()
    for r in responses:
        for lbl in r.get("labels", []):
            label_types[lbl.get("label_type", "UNKNOWN")] += 1
    print(f"\nSpan-level label_type distribution:")
    for lt, cnt in sorted(label_types.items(), key=lambda x: -x[1]):
        print(f"  {lt}: {cnt}")

    # due_to_null and implicit_true flags
    due_to_null_true = sum(
        1 for r in responses
        for lbl in r.get("labels", [])
        if lbl.get("due_to_null") is True
    )
    implicit_true_count = sum(
        1 for r in responses
        for lbl in r.get("labels", [])
        if lbl.get("implicit_true") is True
    )
    print(f"\n  Span flag 'due_to_null==True': {due_to_null_true}")
    print(f"  Span flag 'implicit_true==True': {implicit_true_count}")

    # Span annotation structure summary
    print(f"\nSpan annotation structure (from first record with labels):")
    for r in responses:
        if r.get("labels"):
            lbl = r["labels"][0]
            print(f"  Keys in a label dict: {list(lbl.keys())}")
            print(f"  'start' type: {type(lbl.get('start')).__name__} (character offset)")
            print(f"  'end'   type: {type(lbl.get('end')).__name__}   (character offset)")
            print(f"  'text'  type: {type(lbl.get('text')).__name__}  (the hallucinated span text)")
            print(f"  'label_type': {lbl.get('label_type')!r}")
            break

    # Model distribution
    model_counter = Counter(r.get("model") for r in responses)
    print(f"\nModel distribution:")
    for m, cnt in sorted(model_counter.items(), key=lambda x: -x[1]):
        print(f"  {m}: {cnt}")

    # Split distribution
    split_counter = Counter(r.get("split") for r in responses)
    print(f"\nSplit distribution:")
    for sp, cnt in sorted(split_counter.items()):
        print(f"  {sp}: {cnt}")

    # quality distribution
    quality_counter = Counter(r.get("quality") for r in responses)
    print(f"\n'quality' field distribution:")
    for q, cnt in sorted(quality_counter.items()):
        print(f"  {q}: {cnt}")

    # Where do passages live?
    print(f"\nContext/passages location in source_info.jsonl:")
    for r in sources[:5]:
        si = r.get("source_info", {})
        if isinstance(si, dict) and "passages" in si:
            print(f"  task_type={r['task_type']!r}: source_info['passages'] (string of concatenated passages)")
            break
    for r in sources[:5]:
        si = r.get("source_info", {})
        if isinstance(si, str):
            print(f"  task_type={r['task_type']!r}: source_info is a plain string (article text)")
            break
    print(f"  Join key: source_info.source_id == response.source_id")


# ===========================================================================
# HaluEval
# ===========================================================================

HALUEVAL_HF_REPO = "pminervini/HaluEval"
HALUEVAL_CONFIGS = [
    "qa",
    "qa_samples",
    "dialogue",
    "dialogue_samples",
    "summarization",
    "summarization_samples",
    "general",
]


def explore_halueval():
    from datasets import load_dataset

    _print_section("DATASET: HaluEval")
    print(f"\nSource      : Hugging Face — {HALUEVAL_HF_REPO}")
    print(f"Subsets     : {HALUEVAL_CONFIGS}")

    for config in HALUEVAL_CONFIGS:
        _print_section(f"HaluEval | subset: {config}")
        print(f"\n  HF dataset id : {HALUEVAL_HF_REPO}")
        print(f"  Config name   : {config}")

        try:
            ds = load_dataset(
                HALUEVAL_HF_REPO,
                config,
                cache_dir=str(HALUEVAL_DIR),
                trust_remote_code=False,
            )
        except Exception as exc:
            print(f"  ERROR loading subset '{config}': {exc}")
            continue

        print(f"\n  Available splits: {list(ds.keys())}")

        for split_name, split_ds in ds.items():
            print(f"\n  --- Split: {split_name} ---")
            print(f"  Number of records: {len(split_ds)}")

            if len(split_ds) == 0:
                print("  (empty)")
                continue

            # Field names and types
            first = split_ds[0]
            print(f"\n  Field names and Python types:")
            for k, v in first.items():
                print(f"    {k!r:35s}  {python_type(v)}")

            print(f"\n  First 3 records:")
            for i in range(min(3, len(split_ds))):
                _print_record(split_ds[i], i + 1)

            # Label distributions
            label_fields = [
                f for f in split_ds.column_names
                if "hallucin" in f.lower() or "label" in f.lower()
            ]
            for lf in label_fields:
                vals = split_ds[lf]
                if vals and isinstance(vals[0], str):
                    counter = Counter(vals)
                    print(f"\n  Label distribution for field '{lf}':")
                    for lv, cnt in sorted(counter.items()):
                        print(f"    {lv!r}: {cnt}")
                elif vals and isinstance(vals[0], list):
                    nonempty = sum(1 for v in vals if v)
                    print(f"\n  Field '{lf}' (list): {nonempty}/{len(vals)} records non-empty")

            # Explicit structural notes
            cols = set(split_ds.column_names)
            print(f"\n  Structural notes:")
            if "right_answer" in cols and "hallucinated_answer" in cols:
                print(f"    PAIR structure: each record = (right_answer, hallucinated_answer) + knowledge/question")
            elif "right_response" in cols and "hallucinated_response" in cols:
                print(f"    PAIR structure: each record = (right_response, hallucinated_response) + knowledge/dialogue_history")
            elif "hallucination" in cols:
                val = split_ds[0].get("hallucination", "")
                print(f"    SINGLE-RESPONSE structure: one response + 'hallucination' label (value seen: {val!r})")
            elif "hallucination_spans" in cols:
                print(f"    SINGLE-RESPONSE structure with span annotations ('hallucination_spans' field)")

            knowledge_present = any(f in cols for f in ["knowledge", "document"])
            print(f"    Knowledge/context field present: {knowledge_present}")
            if knowledge_present:
                kf = "knowledge" if "knowledge" in cols else "document"
                sample_k = split_ds[0].get(kf, "")
                print(f"    Knowledge field name: '{kf}', sample length: {len(str(sample_k))} chars")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    explore_ragtruth()
    explore_halueval()
    print(f"\n{_SEP}")
    print("  Exploration complete.")
    print(_SEP)
