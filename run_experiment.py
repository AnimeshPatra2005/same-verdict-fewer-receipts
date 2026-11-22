"""Step 2-4: ask every selected question to every model under four audiences.

Audiences (knowledge only; matched in length and structure):
  control       - the bare question
  novice        - no background in the field
  intermediate  - popular books and podcasts, no formal study
  expert        - ten years of professional practice in the field

Every response ends with BOTTOM LINE / STANCE / CONFIDENCE lines so we can compare runs.

Usage:
  python run_experiment.py prompts [--id 698]              # print the 4 prompts for one question
  python run_experiment.py smoke --model coarena/claude-sonnet-5-5 [--id 698] [--audience novice]
  python run_experiment.py limits                          # Coarena spend today
  python run_experiment.py run --models a,b [--ids 1,2] [--reps 2] [--workers 2]
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from llm import QuotaExhausted, _key, append_jsonl, call, read_jsonl

ROOT = Path(__file__).resolve().parent
QUESTIONS = ROOT / "data" / "selected_questions.csv"
RESPONSES = ROOT / "results" / "responses.jsonl"

AUDIENCES = ["control", "novice", "intermediate", "expert"]

DOMAIN_WORDS = {
    "health": "health and medicine",
    "economics": "economics and finance",
    "science": "science and technology",
    "law": "law and legal matters",
}
EXPERT = {
    "health": "I'm a physician with ten years of clinical practice, and I read the medical research literature regularly as part of my job.",
    "economics": "I'm an economist with ten years of research experience, and I read the economics literature regularly as part of my job.",
    "science": "I'm a research scientist with ten years of experience, and I read the scientific literature regularly as part of my job.",
    "law": "I'm a practising attorney with ten years of experience, and I read case law and statutes regularly as part of my job.",
}

SUFFIX = (
    "At the end of your answer, add these three lines:\n"
    "BOTTOM LINE: one sentence\n"
    "STANCE: Yes / Probably yes / It depends / Probably no / No\n"
    "CONFIDENCE: 0-100 (how confident you are in your bottom line)"
)


def persona(field: str, audience: str) -> str | None:
    if audience == "control":
        return None
    d = DOMAIN_WORDS[field]
    if audience == "novice":
        return f"A bit about me: I don't really know anything about {d}. I've never read up on it or studied it, so most of this is new to me."
    if audience == "intermediate":
        return f"A bit about me: I've read a few popular books about {d} and I follow a couple of podcasts on it, so I know the basics, but I've never studied it formally."
    return "A bit about me: " + EXPERT[field]


def build_prompt(q: dict, audience: str) -> str:
    parts = [p for p in (persona(q["field"], audience), q["question"], SUFFIX) if p]
    return "\n\n".join(parts)


def load_questions() -> list[dict]:
    return list(csv.DictReader(open(QUESTIONS, encoding="utf-8")))


def coarena_limits() -> dict:
    r = requests.get("https://coarena.ai/api/v1/limits",
                     headers={"Authorization": f"Bearer {_key('coarena')}"}, timeout=30)
    try:
        return r.json()
    except ValueError:
        return {"status": r.status_code, "text": r.text[:300]}


def cmd_prompts(a):
    qs = {q["id"]: q for q in load_questions()}
    q = qs[a.id]
    for aud in AUDIENCES:
        print(f"===== {aud} =====\n{build_prompt(q, aud)}\n")


def cmd_limits(a):
    print(json.dumps(coarena_limits(), indent=2))


def cmd_smoke(a):
    qs = {q["id"]: q for q in load_questions()}
    q = qs[a.id]
    prompt = build_prompt(q, a.audience)
    res = call(a.model, prompt)
    rec = {"model": a.model, "id": q["id"], "tier": q["tier"], "field": q["field"], "audience": a.audience,
           "rep": "smoke", "prompt": prompt, **res}
    append_jsonl(ROOT / "results" / "smoke.jsonl", rec)
    if "error" in res:
        print("ERROR:", res["error"])
        return
    print(f"served_model={res['served_model']}  finish={res['finish']}  latency={res['latency_s']}s")
    print("usage:", json.dumps(res["usage"]))
    print("----- response -----")
    print(res["text"])


def cmd_run(a):
    qs = load_questions()
    if a.ids:
        keep = set(a.ids.split(","))
        qs = [q for q in qs if q["id"] in keep]
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    done = {(r["model"], r["id"], r["audience"], r["rep"]) for r in read_jsonl(RESPONSES)
            if "error" not in r and r.get("text")}
    jobs = [(m, q, aud, rep) for rep in range(a.reps) for q in qs for m in models for aud in AUDIENCES
            if (m, q["id"], aud, rep) not in done]
    print(f"{len(jobs)} calls to make ({len(done)} already done)", flush=True)
    stop = {"flag": False}

    def work(job):
        m, q, aud, rep = job
        if stop["flag"]:
            return None
        prompt = build_prompt(q, aud)
        try:
            res = call(m, prompt)
        except QuotaExhausted as e:
            stop["flag"] = True
            return {"quota": str(e)}
        rec = {"model": m, "id": q["id"], "tier": q["tier"], "field": q["field"], "audience": aud,
               "rep": rep, "prompt": prompt, "ts": time.time(), **res}
        append_jsonl(RESPONSES, rec)
        return rec

    n_err = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            if rec and "quota" in rec:
                print("STOPPING:", rec["quota"], flush=True)
            elif rec and "error" in rec:
                n_err += 1
                print(f"  error {rec['model']} q{rec['id']} {rec['audience']}: {rec['error'][:160]}", flush=True)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} processed, {n_err} errors", flush=True)


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("prompts"); x.add_argument("--id", default="698"); x.set_defaults(f=cmd_prompts)
    x = s.add_parser("limits"); x.set_defaults(f=cmd_limits)
    x = s.add_parser("smoke"); x.add_argument("--model", required=True); x.add_argument("--id", default="698")
    x.add_argument("--audience", default="novice"); x.set_defaults(f=cmd_smoke)
    x = s.add_parser("run"); x.add_argument("--models", required=True); x.add_argument("--ids", default="")
    x.add_argument("--reps", type=int, default=2); x.add_argument("--workers", type=int, default=2)
    x.set_defaults(f=cmd_run)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
