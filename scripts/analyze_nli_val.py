"""
scripts/analyze_nli_val.py
--------------------------
Analysis of NLI validation runs on RAGTruth.
Uses cached probabilities exclusively; does NOT perform model inference.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

REPO_ROOT = Path("/home/srishti-rawat/RAGuard")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score

from claims.splitter import split_claims
from detectors.nli import _window_text
from eval.datasets import load_ragtruth
from eval.harness import load_and_verify_split_ids
from eval.metrics import grouped_bootstrap_ci

REPO_ROOT = Path("/home/srishti-rawat/RAGuard")


def load_run_predictions(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Load predictions.jsonl keyed by example id."""
    preds: dict[str, dict[str, Any]] = {}
    with open(run_dir / "predictions.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                preds[rec["id"]] = rec
    return preds


def load_run_config(run_dir: Path) -> dict[str, Any]:
    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def compute_auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    y_s = np.asarray(scores, dtype=float)
    y_t = np.asarray(labels, dtype=int)
    if len(np.unique(y_t)) < 2:
        return None
    if len(np.unique(y_s)) == 1:
        return 0.5
    try:
        return float(roc_auc_score(y_t, y_s))
    except Exception:
        return None


def paired_grouped_bootstrap_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    labels: Sequence[int],
    groups: Sequence[str],
    n: int = 1000,
    seed: int = 42,
) -> tuple[float, tuple[Optional[float], Optional[float]]]:
    """
    Compute point estimate and 95% CI of AUC(scores_a) - AUC(scores_b)
    using grouped bootstrap resampling of groups (source_id).
    """
    y_sa = np.asarray(scores_a, dtype=float)
    y_sb = np.asarray(scores_b, dtype=float)
    y_true = np.asarray(labels, dtype=int)
    y_groups = np.asarray(groups)

    auc_a = compute_auc(y_sa, y_true)
    auc_b = compute_auc(y_sb, y_true)
    diff_pt = (auc_a - auc_b) if (auc_a is not None and auc_b is not None) else 0.0

    unique_groups = np.array(sorted(list(set(y_groups))))
    group_indices = {g: np.where(y_groups == g)[0] for g in unique_groups}

    rng = np.random.default_rng(seed)
    deltas: list[float] = []

    for _ in range(n):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sample_idx = np.concatenate([group_indices[g] for g in sampled_groups])

        sub_sa = y_sa[sample_idx]
        sub_sb = y_sb[sample_idx]
        sub_y = y_true[sample_idx]

        val_a = compute_auc(sub_sa, sub_y)
        val_b = compute_auc(sub_sb, sub_y)

        if val_a is not None and val_b is not None:
            deltas.append(val_a - val_b)

    if not deltas:
        return diff_pt, (None, None)

    ci_lower, ci_upper = np.percentile(deltas, [2.5, 97.5])
    return diff_pt, (float(ci_lower), float(ci_upper))


def main():
    print("=" * 80)
    print("  RAGuard: NLI Validation Runs Empirical Analysis")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Part 1: Confirm runs and print config metadata
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  Part 1: Run Metadata Confirmation (from config.json)")
    print("=" * 80)

    nli_runs = [
        ("max", REPO_ROOT / "runs/20261114_062508_nli_val"),
        ("mean", REPO_ROOT / "runs/20261114_062529_nli_val"),
        ("noisy_or", REPO_ROOT / "runs/20261114_062548_nli_val"),
    ]
    length_run = REPO_ROOT / "runs/20261111_165842_length_val"

    for expected_agg, run_dir in nli_runs:
        cfg = load_run_config(run_dir)
        actual_agg = cfg.get("aggregation")
        trunc_rate = cfg.get("truncation_rate")
        hit_rate = cfg.get("cache_hit_rate")
        model_name = cfg.get("model_name")
        print(f"Run Directory: {run_dir.name}")
        print(f"  Model Name       : {model_name}")
        print(f"  Expected Agg     : {expected_agg}")
        print(f"  Config Agg       : {actual_agg} (Matches: {expected_agg == actual_agg})")
        print(f"  Truncation Rate  : {trunc_rate:.4f} ({trunc_rate * 100:.2f}%)")
        print(f"  Cache Hit Rate   : {hit_rate:.4f} ({hit_rate * 100:.2f}%)")
        print()

    # Load predictions
    preds_max = load_run_predictions(nli_runs[0][1])
    preds_len = load_run_predictions(length_run)

    # Load validation split examples to align metadata
    split_ids, _ = load_and_verify_split_ids(REPO_ROOT / "data/processed/splits", "val")
    split_set = set(split_ids)
    all_examples = load_ragtruth(REPO_ROOT / "data/raw/ragtruth")
    val_examples = [ex for ex in all_examples if ex.id in split_set]
    ex_by_id = {ex.id: ex for ex in val_examples}

    # Verify example alignment
    common_ids = [ex.id for ex in val_examples if ex.id in preds_max and ex.id in preds_len]
    assert len(common_ids) == len(val_examples) == 1449, f"Mismatch in count: {len(common_ids)}"

    # -----------------------------------------------------------------------
    # Part 2: Grouped bootstrap ROC-AUC & Paired difference
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("  Part 2: Grouped Bootstrap ROC-AUC (1000 draws, seed 42) & Paired Delta")
    print("=" * 80)

    # Prepare data arrays
    subsets = ["QA", "Summary"]
    models = sorted(list({ex_by_id[i].meta.get("model", "unknown") for i in common_ids}))

    def _eval_group(ids: list[str]):
        labels = [1 if ex_by_id[i].label_response_level else 0 for i in ids]
        groups = [ex_by_id[i].group_id for i in ids]
        scores_nli = [preds_max[i]["score"] for i in ids]
        scores_len = [preds_len[i]["score"] for i in ids]

        # Single bootstrap CI helper
        def _auc_fn(s, y):
            return compute_auc(s, y)

        auc_nli = compute_auc(scores_nli, labels)
        auc_len = compute_auc(scores_len, labels)

        ci_nli = grouped_bootstrap_ci(_auc_fn, scores_nli, labels, groups, n=1000, seed=42)
        ci_len = grouped_bootstrap_ci(_auc_fn, scores_len, labels, groups, n=1000, seed=42)

        diff, ci_diff = paired_grouped_bootstrap_ci(scores_nli, scores_len, labels, groups, n=1000, seed=42)

        return {
            "n": len(ids),
            "prevalence": np.mean(labels),
            "auc_nli": auc_nli,
            "ci_nli": ci_nli,
            "auc_len": auc_len,
            "ci_len": ci_len,
            "diff": diff,
            "ci_diff": ci_diff,
        }

    # Overall
    res_overall = _eval_group(common_ids)
    print("\n--- OVERALL (Val, N=1449) ---")
    print(f"  NLI (max) ROC-AUC : {res_overall['auc_nli']:.4f} (95% CI: [{res_overall['ci_nli'][0]:.4f}, {res_overall['ci_nli'][1]:.4f}])")
    print(f"  Length   ROC-AUC : {res_overall['auc_len']:.4f} (95% CI: [{res_overall['ci_len'][0]:.4f}, {res_overall['ci_len'][1]:.4f}])")
    print(f"  Paired Difference : {res_overall['diff']:+.4f} (95% CI: [{res_overall['ci_diff'][0]:+.4f}, {res_overall['ci_diff'][1]:+.4f}])")

    # Per subset
    print("\n--- PER SUBSET ---")
    for sub in subsets:
        sub_ids = [i for i in common_ids if ex_by_id[i].subset == sub]
        r = _eval_group(sub_ids)
        print(f"[{sub:8s}] N={r['n']:4d} | Prev={r['prevalence']:.4f}")
        print(f"  NLI (max) ROC-AUC : {r['auc_nli']:.4f} (95% CI: [{r['ci_nli'][0]:.4f}, {r['ci_nli'][1]:.4f}])")
        print(f"  Length   ROC-AUC : {r['auc_len']:.4f} (95% CI: [{r['ci_len'][0]:.4f}, {r['ci_len'][1]:.4f}])")
        print(f"  Paired Delta (NLI - Len): {r['diff']:+.4f} (95% CI: [{r['ci_diff'][0]:+.4f}, {r['ci_diff'][1]:+.4f}])")

    # Per model
    print("\n--- PER META['MODEL'] ---")
    print(f"{'Model':20s} | {'N':4s} | {'Prev':6s} | {'NLI Max AUC (95% CI)':28s} | {'Length AUC (95% CI)':28s}")
    print("-" * 95)
    for mod in models:
        mod_ids = [i for i in common_ids if ex_by_id[i].meta.get("model") == mod]
        r = _eval_group(mod_ids)
        ci_n = f"[{r['ci_nli'][0]:.4f}, {r['ci_nli'][1]:.4f}]" if r['ci_nli'][0] is not None else "N/A"
        ci_l = f"[{r['ci_len'][0]:.4f}, {r['ci_len'][1]:.4f}]" if r['ci_len'][0] is not None else "N/A"
        print(f"{mod:20s} | {r['n']:4d} | {r['prevalence']:6.4f} | {r['auc_nli']:.4f} {ci_n:20s} | {r['auc_len']:.4f} {ci_l:20s}")

    # -----------------------------------------------------------------------
    # Part 3: Score distribution for NLI max on val
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  Part 3: Score Distribution for NLI (max) on Val")
    print("=" * 80)

    nli_max_scores = np.array([preds_max[i]["score"] for i in common_ids])
    unique_vals = np.unique(nli_max_scores)
    frac_ge_99 = np.mean(nli_max_scores >= 0.99)
    frac_ge_999 = np.mean(nli_max_scores >= 0.999)

    print(f"Total Examples Evaluated           : {len(nli_max_scores)}")
    print(f"Number of Unique Score Values      : {len(unique_vals)}")
    print(f"Fraction with Risk >= 0.99 (99%)   : {frac_ge_99:.4f} ({frac_ge_99 * 100:.2f}%) [{np.sum(nli_max_scores >= 0.99)} / {len(nli_max_scores)}]")
    print(f"Fraction with Risk >= 0.999 (99.9%): {frac_ge_999:.4f} ({frac_ge_999 * 100:.2f}%) [{np.sum(nli_max_scores >= 0.999)} / {len(nli_max_scores)}]")

    # Percentiles
    pcts = [0, 25, 50, 75, 90, 95, 99, 100]
    pct_vals = np.percentile(nli_max_scores, pcts)
    print("Score Percentiles:")
    for p, v in zip(pcts, pct_vals):
        print(f"  p{p:02d}: {v:.6f}")

    # -----------------------------------------------------------------------
    # Part 4: Re-score from cached probabilities (Probability vs Logit version)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  Part 4: Re-scoring from Cached Probabilities (Prob vs Logit of Support)")
    print("=" * 80)

    cache_db = REPO_ROOT / "data/processed/cache/cross-encoder__nli-deberta-v3-base.db"
    assert cache_db.exists(), f"Cache DB missing: {cache_db}"
    conn = sqlite3.connect(cache_db)

    # Pre-load cache keys into memory dict for ultra-fast lookup
    print("Loading SQLite cache table into memory...")
    cur = conn.cursor()
    cur.execute("SELECT key, p_ent, p_neu, p_con FROM nli_cache")
    cache_dict = {row[0]: (row[1], row[2], row[3]) for row in cur.fetchall()}
    conn.close()
    print(f"Loaded {len(cache_dict)} entries from cache.")

    import hashlib
    def get_cached(premise: str, hypothesis: str) -> tuple[float, float, float]:
        raw = f"cross-encoder/nli-deberta-v3-base\x00{premise}\x00{hypothesis}"
        k = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if k not in cache_dict:
            raise KeyError(f"Cache miss for pair! premise={premise[:40]}..., hyp={hypothesis[:40]}...")
        return cache_dict[k]

    # Re-compute per example:
    # 1. prob version: response_risk = max_claim (1 - max_window p_ent)
    # 2. logit version:
    #    support = max_window p_ent
    #    logit_support = log(p / (1 - p)) clipped to [eps, 1 - eps]
    #    risk_logit = max_claim (-logit_support) = - min_claim (logit_support)
    eps = 1e-6
    scores_prob_recomputed: dict[str, float] = {}
    scores_logit: dict[str, float] = {}
    claim_details: dict[str, list[dict[str, Any]]] = {}

    for ex in val_examples:
        claims = split_claims(ex.response)
        if not claims:
            scores_prob_recomputed[ex.id] = 0.0
            scores_logit[ex.id] = -50.0  # arbitrary low risk
            claim_details[ex.id] = []
            continue

        windows = _window_text(ex.context, 180, 90)
        ex_claims = []
        claim_risks_prob = []
        claim_risks_logit = []

        for c in claims:
            # find max p_ent across windows
            max_p_ent = -1.0
            for w in windows:
                p_ent, p_neu, p_con = get_cached(w, c.text)
                if p_ent > max_p_ent:
                    max_p_ent = p_ent

            # probability risk
            c_risk_prob = 1.0 - max_p_ent
            claim_risks_prob.append(c_risk_prob)

            # logit version of support score
            p_clipped = min(max(max_p_ent, eps), 1.0 - eps)
            log_odds_support = math.log(p_clipped / (1.0 - p_clipped))
            # risk in logit space is negative support log-odds
            c_risk_logit = -log_odds_support
            claim_risks_logit.append(c_risk_logit)

            ex_claims.append({
                "claim_text": c.text,
                "start": c.start,
                "end": c.end,
                "p_ent": max_p_ent,
                "risk_prob": c_risk_prob,
                "log_odds_support": log_odds_support,
                "risk_logit": c_risk_logit,
            })

        scores_prob_recomputed[ex.id] = float(max(claim_risks_prob))
        scores_logit[ex.id] = float(max(claim_risks_logit))
        claim_details[ex.id] = ex_claims

    # Verify recomputed matches predictions.jsonl
    diffs = [abs(scores_prob_recomputed[i] - preds_max[i]["score"]) for i in common_ids]
    max_diff = max(diffs)
    print(f"Max difference between recomputed prob scores and predictions.jsonl: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Recomputed prob scores diverged from stored predictions! max_diff={max_diff}"

    # Evaluate Probability vs Logit version
    def _eval_prob_vs_logit(ids: list[str]):
        labels = [1 if ex_by_id[i].label_response_level else 0 for i in ids]
        groups = [ex_by_id[i].group_id for i in ids]
        s_prob = [scores_prob_recomputed[i] for i in ids]
        s_logit = [scores_logit[i] for i in ids]

        auc_p = compute_auc(s_prob, labels)
        ci_p = grouped_bootstrap_ci(compute_auc, s_prob, labels, groups, n=1000, seed=42)

        auc_l = compute_auc(s_logit, labels)
        ci_l = grouped_bootstrap_ci(compute_auc, s_logit, labels, groups, n=1000, seed=42)

        diff, ci_d = paired_grouped_bootstrap_ci(s_logit, s_prob, labels, groups, n=1000, seed=42)

        return {
            "n": len(ids),
            "auc_prob": auc_p,
            "ci_prob": ci_p,
            "auc_logit": auc_l,
            "ci_logit": ci_l,
            "diff_logit_minus_prob": diff,
            "ci_diff": ci_d,
        }

    p_vs_l_all = _eval_prob_vs_logit(common_ids)
    print("\n--- OVERALL (Val, N=1449) ---")
    print(f"  Probability Version ROC-AUC : {p_vs_l_all['auc_prob']:.4f} (95% CI: [{p_vs_l_all['ci_prob'][0]:.4f}, {p_vs_l_all['ci_prob'][1]:.4f}])")
    print(f"  Logit Version       ROC-AUC : {p_vs_l_all['auc_logit']:.4f} (95% CI: [{p_vs_l_all['ci_logit'][0]:.4f}, {p_vs_l_all['ci_logit'][1]:.4f}])")
    print(f"  Delta (Logit - Prob)        : {p_vs_l_all['diff_logit_minus_prob']:+.4f} (95% CI: [{p_vs_l_all['ci_diff'][0]:+.4f}, {p_vs_l_all['ci_diff'][1]:+.4f}])")

    for sub in subsets:
        sub_ids = [i for i in common_ids if ex_by_id[i].subset == sub]
        r = _eval_prob_vs_logit(sub_ids)
        print(f"\n--- SUBSET: {sub} (N={r['n']}) ---")
        print(f"  Probability Version ROC-AUC : {r['auc_prob']:.4f} (95% CI: [{r['ci_prob'][0]:.4f}, {r['ci_prob'][1]:.4f}])")
        print(f"  Logit Version       ROC-AUC : {r['auc_logit']:.4f} (95% CI: [{r['ci_logit'][0]:.4f}, {r['ci_logit'][1]:.4f}])")
        print(f"  Delta (Logit - Prob)        : {r['diff_logit_minus_prob']:+.4f} (95% CI: [{r['ci_diff'][0]:+.4f}, {r['ci_diff'][1]:+.4f}])")

    # -----------------------------------------------------------------------
    # Part 5: Show-errors 5 (reproduced directly from predictions + examples)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  Part 5: Harness --show-errors 5 (Console Only, Val QA)")
    print("=" * 80)

    qa_ids = [i for i in common_ids if ex_by_id[i].subset == "QA"]
    qa_preds = [preds_max[i] for i in qa_ids]
    threshold_max = load_run_config(nli_runs[0][1])["threshold"]

    not_hal_qa = sorted([p for p in qa_preds if not p["label"]], key=lambda x: x["score"], reverse=True)[:5]
    hal_qa = sorted([p for p in qa_preds if p["label"]], key=lambda x: x["score"])[:5]

    print(f"Threshold (Tuned on Val): {threshold_max:.4f}")
    print(f"\n--- TOP-5 HIGHEST-RISK NON-HALLUCINATED (False Positives at threshold {threshold_max:.4f}) ---")
    for rank, p in enumerate(not_hal_qa, 1):
        ex = ex_by_id[p["id"]]
        print(f"\n[{rank}] ID={p['id']} | score={p['score']:.4f} | label={p['label']} | model={p['model']}")
        if not ex.spans:
            print("  (no annotated spans)")
        else:
            for s_idx, span in enumerate(ex.spans, 1):
                trunc_span = span.text[:150] + ("..." if len(span.text) > 150 else "")
                print(f"  Span {s_idx}: label_type={span.label_type!r} | {trunc_span!r}")

    print(f"\n--- TOP-5 LOWEST-RISK HALLUCINATED (False Negatives at threshold {threshold_max:.4f}) ---")
    for rank, p in enumerate(hal_qa, 1):
        ex = ex_by_id[p["id"]]
        print(f"\n[{rank}] ID={p['id']} | score={p['score']:.4f} | label={p['label']} | model={p['model']}")
        if not ex.spans:
            print("  (no annotated spans)")
        else:
            for s_idx, span in enumerate(ex.spans, 1):
                trunc_span = span.text[:150] + ("..." if len(span.text) > 150 else "")
                print(f"  Span {s_idx}: label_type={span.label_type!r} | {trunc_span!r}")

    # -----------------------------------------------------------------------
    # Part 6: 15 val QA responses with highest NLI-max risk that are NOT hallucinated
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  Part 6: Top-15 Val QA Highest-Risk NON-Hallucinated Responses")
    print("  (Showing top-risk claim truncated to 150 chars and entailment probability)")
    print("=" * 80)

    # Non-hallucinated QA instances sorted by descending NLI max risk
    all_not_hal_qa = [i for i in qa_ids if not ex_by_id[i].label_response_level]
    all_not_hal_qa_sorted = sorted(all_not_hal_qa, key=lambda i: preds_max[i]["score"], reverse=True)[:15]

    for rank, ex_id in enumerate(all_not_hal_qa_sorted, 1):
        ex = ex_by_id[ex_id]
        score = preds_max[ex_id]["score"]
        claims_info = claim_details[ex_id]

        if claims_info:
            # top risk claim is the one with highest risk (lowest p_ent)
            top_claim = max(claims_info, key=lambda c: c["risk_prob"])
            claim_text = top_claim["claim_text"]
            trunc_claim = claim_text[:150] + ("..." if len(claim_text) > 150 else "")
            p_ent = top_claim["p_ent"]
            c_risk = top_claim["risk_prob"]
        else:
            trunc_claim = "(no claims found)"
            p_ent = 1.0
            c_risk = 0.0

        print(f"\n[{rank:02d}] ID: {ex_id} | Model: {ex.meta.get('model')} | Response Risk: {score:.4f}")
        print(f"     Entailment Probability: {p_ent:.6f} (Claim Risk: {c_risk:.4f})")
        print(f"     Top-Risk Claim        : {trunc_claim}")


if __name__ == "__main__":
    main()
