"""Step 6: analysis.

Per response we have, from the response itself: STANCE (5-point), CONFIDENCE (0-100), length.
From each grader: dispute awareness (DebateQA DA, 0/1), perspective coverage (share of the
human-annotated perspectives mentioned), evidence strength, named sources, verify advice,
caveats, hedges. Grader scores are averaged across the two graders for the main analysis;
grader agreement is reported separately.

Main comparisons (paired by model x question x rep): novice minus expert, and each audience
minus control. Noise baseline: rep 0 vs rep 1 within the same audience. Significance: sign-flip
permutation test on paired differences, 95% bootstrap CIs.

Usage: python analyze.py [--out results/analysis]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESPONSES = ROOT / "results" / "responses.jsonl"
GRADES = ROOT / "results" / "grades.jsonl"
AUD = ["control", "novice", "intermediate", "expert"]
TIERS = ["debated", "consensus_contested", "trivia"]

STANCE_MAP = {"yes": 2, "probably yes": 1, "it depends": 0, "depends": 0, "probably no": -1, "no": -2}
STANCE_RE = re.compile(r"STANCE\W*\s*[:\-]\s*\**\s*([A-Za-z ]+)", re.I)
CONF_RE = re.compile(r"CONFIDENCE\W*\s*[:\-]\s*\**\s*(\d{1,3})", re.I)


def parse_stance(t):
    m = STANCE_RE.findall(t or "")
    if not m:
        return np.nan
    s = m[-1].strip().lower().rstrip(".").strip()
    for k in sorted(STANCE_MAP, key=len, reverse=True):
        if s.startswith(k):
            return STANCE_MAP[k]
    return np.nan


def parse_conf(t):
    m = CONF_RE.findall(t or "")
    if not m:
        return np.nan
    v = int(m[-1])
    return v if 0 <= v <= 100 else np.nan


def load():
    rows = []
    for l in open(RESPONSES, encoding="utf-8"):
        r = json.loads(l)
        if "error" in r or not r.get("text"):
            continue
        rows.append({"model": r["model"].split("/", 1)[1], "id": str(r["id"]), "tier": r["tier"],
                     "field": r["field"], "audience": r["audience"], "rep": r["rep"],
                     "stance": parse_stance(r["text"]), "confidence": parse_conf(r["text"]),
                     "words": len(r["text"].split())})
    df = pd.DataFrame(rows).drop_duplicates(["model", "id", "audience", "rep"], keep="last")
    df["decisiveness"] = df["stance"].abs()
    df["says_depends"] = (df["stance"] == 0).astype(float).where(df["stance"].notna())

    g = []
    for l in open(GRADES, encoding="utf-8"):
        x = json.loads(l)
        rb = x.get("rubric") or {}
        if x.get("da") is None or not rb:
            continue
        cov = rb.get("perspectives_covered") or []
        try:
            cov = len({int(c) for c in cov if 1 <= int(c) <= x["n_perspectives"]}) / x["n_perspectives"]
        except (TypeError, ValueError):
            cov = np.nan
        g.append({"model": x["model"].split("/", 1)[1], "id": str(x["id"]), "audience": x["audience"],
                  "rep": x["rep"], "grader": x["grader"].split("/", 1)[1], "dispute_aware": x["da"],
                  "perspective_cov": cov,
                  **{k: pd.to_numeric(rb.get(k), errors="coerce") for k in
                     ["evidence_strength", "specific_sources", "verify_advice", "caveats", "hedges"]}})
    gd = pd.DataFrame(g).drop_duplicates(["model", "id", "audience", "rep", "grader"], keep="last")
    return df, gd


GRADED = ["dispute_aware", "perspective_cov", "evidence_strength", "specific_sources", "verify_advice",
          "caveats", "hedges"]
METRICS = ["stance", "decisiveness", "says_depends", "confidence", "words"] + GRADED


def paired(df, metric, a, b, rng, n_perm=5000):
    piv = df.pivot_table(index=["model", "id", "rep"], columns="audience", values=metric)
    if a not in piv or b not in piv:
        return None
    d = (piv[a] - piv[b]).dropna().values
    if len(d) < 5:
        return None
    boots = rng.choice(d, (2000, len(d))).mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    signs = rng.choice([-1, 1], (n_perm, len(d)))
    null = (signs * np.abs(d)).mean(1)
    p = (np.sum(np.abs(null) >= abs(d.mean())) + 1) / (n_perm + 1)
    return {"n": len(d), "mean_diff": d.mean(), "ci_lo": lo, "ci_hi": hi, "p": p,
            "share_a_higher": np.mean(d > 0), "share_b_higher": np.mean(d < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "analysis"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)

    df, gd = load()
    gmean = gd.groupby(["model", "id", "audience", "rep"])[GRADED].mean().reset_index()
    full = df.merge(gmean, on=["model", "id", "audience", "rep"], how="left")
    full.to_csv(out / "items.csv", index=False)
    print(f"{len(df)} responses, {len(gd)} grader records, models: {sorted(df['model'].unique())}")
    print(f"parse failures: stance {df['stance'].isna().sum()}, confidence {df['confidence'].isna().sum()}")

    # 1. means by tier x audience (pooled over models) and by model
    tab = full.groupby(["tier", "audience"])[METRICS].mean().reindex(
        [(t, x) for t in TIERS for x in AUD if (t, x) in set(zip(full["tier"], full["audience"]))])
    print("\n=== Means by tier x audience (all models) ===")
    print(tab.to_string())
    tab.to_csv(out / "means_tier_audience.csv")
    tabm = full.groupby(["model", "tier", "audience"])[METRICS].mean()
    tabm.to_csv(out / "means_model_tier_audience.csv")

    # 2. paired novice - expert (and vs control) per tier
    res = []
    for tier in TIERS + ["ALL"]:
        sub = full if tier == "ALL" else full[full["tier"] == tier]
        for (x, y) in [("novice", "expert"), ("novice", "control"), ("expert", "control"),
                       ("intermediate", "expert")]:
            for m in METRICS:
                r = paired(sub, m, x, y, rng)
                if r:
                    res.append({"tier": tier, "contrast": f"{x}-{y}", "metric": m, **r})
    res = pd.DataFrame(res)
    res.to_csv(out / "paired_contrasts.csv", index=False)
    print("\n=== Novice minus expert (paired; p = sign-flip permutation) ===")
    print(res[res["contrast"] == "novice-expert"][["tier", "metric", "n", "mean_diff", "ci_lo", "ci_hi", "p",
                                                    "share_a_higher", "share_b_higher"]].to_string(index=False))

    # 3. per model, novice - expert on the key metrics, debated tier
    rows = []
    for mdl in sorted(full["model"].unique()):
        for m in ["decisiveness", "confidence", "dispute_aware", "perspective_cov", "evidence_strength",
                  "specific_sources", "words"]:
            r = paired(full[(full["model"] == mdl) & (full["tier"] != "trivia")], m, "novice", "expert", rng)
            if r:
                rows.append({"model": mdl, "metric": m, **r})
    pm = pd.DataFrame(rows)
    pm.to_csv(out / "per_model_novice_expert.csv", index=False)
    print("\n=== Per model, novice minus expert (debated + contested) ===")
    print(pm[["model", "metric", "n", "mean_diff", "ci_lo", "ci_hi", "p"]].to_string(index=False))

    # 4. noise baseline: rep0 vs rep1 within audience, compared to novice vs expert within rep
    if full["rep"].nunique() > 1:
        nb = []
        for m in ["stance", "confidence", "dispute_aware", "perspective_cov"]:
            piv = full.pivot_table(index=["model", "id", "audience"], columns="rep", values=m)
            within = (piv[0] - piv[1]).abs().dropna()
            piv2 = full.pivot_table(index=["model", "id", "rep"], columns="audience", values=m)
            between = (piv2["novice"] - piv2["expert"]).abs().dropna()
            nb.append({"metric": m, "mean_abs_rep_to_rep": within.mean(),
                       "mean_abs_novice_vs_expert": between.mean(),
                       "share_changed_rep_to_rep": (within > 0).mean(),
                       "share_changed_novice_vs_expert": (between > 0).mean()})
        nb = pd.DataFrame(nb)
        nb.to_csv(out / "noise_baseline.csv", index=False)
        print("\n=== Noise baseline (same audience, two runs) vs audience effect ===")
        print(nb.to_string(index=False))

    # 5. grader agreement
    if gd["grader"].nunique() == 2:
        g1, g2 = sorted(gd["grader"].unique())
        pv = gd.pivot_table(index=["model", "id", "audience", "rep"], columns="grader",
                            values=["dispute_aware", "evidence_strength", "specific_sources", "perspective_cov"])
        ag = {}
        for m in ["dispute_aware", "evidence_strength", "specific_sources"]:
            x = pv[m].dropna()
            ag[m] = float((x[g1] == x[g2]).mean())
        x = pv["perspective_cov"].dropna()
        ag["perspective_cov_corr"] = float(np.corrcoef(x[g1], x[g2])[0, 1]) if len(x) > 2 else np.nan
        print(f"\n=== Grader agreement ({g1} vs {g2}) ===")
        print(json.dumps(ag, indent=2))
        (out / "grader_agreement.json").write_text(json.dumps(ag, indent=2))

    # 6. figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [("decisiveness", "Decisiveness of stance (0-2)"), ("confidence", "Stated confidence"),
            ("dispute_aware", "Says it's debated (share)"), ("perspective_cov", "Perspectives covered (share)")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    x = np.arange(len(AUD))
    tiers_present = [t for t in TIERS if t in set(full["tier"])]
    w = 0.8 / len(tiers_present)
    for ax, (m, title) in zip(axes, show):
        for i, t in enumerate(tiers_present):
            s = full[full["tier"] == t].groupby("audience")[m].mean().reindex(AUD)
            ax.bar(x + i * w - 0.4 + w / 2, s.values, w, label=t)
        ax.set_xticks(x, AUD, fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_audience_by_tier.png", dpi=160)
    print(f"\nFigures and tables in {out}")


if __name__ == "__main__":
    main()
