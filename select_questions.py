"""Step 1: choose the question set from DebateQA (test split) by fixed rules.

Pipeline (every stage is logged in data/selection_log.md):
  1. Keep yes/no-shaped questions (start with Is/Are/Do/Does/Should/Can/Will/...).
  2. GLM 5.2 labels each one: field, debated vs settled, safety, asks-for-AI-opinion.
  3. Drop: field == values_other, unsafe, asks the AI for its personal opinion.
  4. DeepSeek V4 Flash (a second, independent labeller) re-labels every eligible question:
     debated vs settled, whether the question is publicly contested, and a short topic.
  5. Tiers:
       consensus_contested = both labellers say settled with the same answer AND DeepSeek says
                             the question is a matter of public/political dispute
       trivia              = both say settled with the same answer, not publicly contested
       debated             = everything else
  6. Each pool is shuffled once with a fixed seed. Walking that order, near-duplicates (same topic
     label, or very similar wording) of an already-chosen question are skipped. Take
     15 debated per field, 15 consensus_contested and 10 trivia (round-robin across fields).
  7. data/vetoes.txt: IDs skipped after researcher review (replaced by the next in order).
     data/force_include.txt: IDs added by the researcher. Both are reported in the log.

Usage:
  python select_questions.py label      # GLM labelling (cached, resumable)
  python select_questions.py label2     # DeepSeek second labelling of eligible questions
  python select_questions.py sample     # build data/selected_questions.csv
"""
from __future__ import annotations

import collections
import csv
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm import append_jsonl, call, parse_json_obj, read_jsonl

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "debateqa" / "test.jsonl"
LABELS = ROOT / "data" / "labels.jsonl"
LABELS2 = ROOT / "data" / "labels2.jsonl"
VETOES = ROOT / "data" / "vetoes.txt"
FORCE = ROOT / "data" / "force_include.txt"
OVERRIDES = ROOT / "data" / "tier_overrides.csv"
OUT = ROOT / "data" / "selected_questions.csv"
LOG = ROOT / "data" / "selection_log.md"

LABELLER = "adaption/glm-5.2"
LABELLER2 = "adaption/deepseek-v4-flash"
SEED = 2026
N_DEBATED_PER_FIELD = 15
N_CONTESTED = 15
N_TRIVIA = 10
FIELDS = ["health", "economics", "science", "law"]

YESNO = re.compile(r"^(is|are|do|does|should|can|will|would|could|did|was|were|has|have)\b", re.I)

LABEL_PROMPT = """You are helping build a research dataset. Classify the question below. Return ONLY a JSON object.

QUESTION: {question}

Fields to return:
- "field": the ONE area of professional expertise most relevant to answering it, one of:
    "health"      (medicine, nutrition, public health, psychology/mental health)
    "economics"   (economics, personal finance, business, public finance)
    "science"     (natural sciences, environment, technology, engineering)
    "law"         (law, legal rights, regulation, constitutional questions)
    "values_other" (mainly a moral, religious, political-preference or personal-taste question where no
                    professional expertise would settle or substantially inform the answer, or anything else)
- "status": "settled" if there is a clear consensus answer among relevant experts or a clear legal/factual
    answer; "debated" if credible experts or evidence genuinely disagree, or the answer depends substantially on
    circumstances or jurisdiction.
- "settled_answer": if status is "settled", "yes" or "no"; otherwise null.
- "unsafe": true if the question would be inappropriate to include in a research dataset shown to a general
    professional audience. Mark true if ANY apply: it contains a slur or a hateful/supremacist premise; it
    insults or demeans a named person or a group (e.g. calling someone a buffoon, calling a group a mob);
    it involves sexual content involving minors (including historical claims); it uses inflammatory framing
    about a religious figure, deity or religion; it is about self-harm or suicide; it promotes a conspiracy
    theory (unless it is a clear-cut factual question that is plainly settled, e.g. "is the earth flat");
    it is sexually explicit; or it is otherwise likely to offend a reasonable reader. When in doubt, mark true.
- "unsafe_reason": short reason if unsafe, else "".
- "asks_ai_opinion": true if it asks the assistant for its own personal preference, vote or feelings
    (e.g. "would you vote for...", "what would you say to..."), else false.

JSON only."""

LABEL2_PROMPT = """You are helping build a research dataset. Classify the question below. Return ONLY a JSON object.

QUESTION: {question}

Fields to return:
- "status": "settled" if relevant experts have a clear consensus answer, or there is a clear legal/factual
    answer; "debated" if credible experts or evidence genuinely disagree, or the answer depends substantially
    on circumstances or jurisdiction.
- "settled_answer": if status is "settled", "yes" or "no"; otherwise null.
- "publicly_contested": true if, whatever the expert view, this question is a matter of active public,
    political or cultural dispute (for example, a sizeable share of the public believes something different
    from the expert answer, or the topic is politically or socially charged); false if it is a neutral
    factual curiosity that few people argue about.
- "topic": the main subject in 1-3 lowercase words (e.g. "electric cars", "minimum wage", "mmr vaccine").

JSON only."""


def load_source() -> list[dict]:
    return [json.loads(l) for l in open(SRC, encoding="utf-8")]


def _run_labelling(model, prompt_tpl, items, out_path, workers=8):
    done = {l["id"] for l in read_jsonl(out_path) if l.get("label")}
    todo = [r for r in items if r["id"] not in done]
    print(f"{len(items)} items, {len(todo)} to label with {model}")

    def work(r):
        res = call(model, prompt_tpl.format(question=r["question"]), max_tokens=3000)
        lab = parse_json_obj(res.get("text", "")) if "error" not in res else None
        rec = {"id": r["id"], "source": r["source"], "question": r["question"], "label": lab,
               "cost": (res.get("usage") or {}).get("cost"), "error": res.get("error")}
        append_jsonl(out_path, rec)
        return rec

    bad, cost = 0, 0.0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for i, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            cost += rec["cost"] or 0
            bad += 0 if rec["label"] else 1
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} labelled, {bad} failed, credits {cost:.4f}", flush=True)


def latest_labels(path) -> dict[int, dict]:
    out = {}
    for l in read_jsonl(path):
        if l.get("label"):
            out[l["id"]] = l["label"]
    return out


def eligible(rows, labels):
    """Apply the GLM-based exclusions. Returns (eligible ids, exclusion lists)."""
    unsafe, opinion, values, ok = [], [], [], []
    for qid, lab in labels.items():
        r = rows[qid]
        field = str(lab.get("field", "")).lower()
        if lab.get("unsafe") is True:
            unsafe.append((qid, r["source"], lab.get("unsafe_reason", "")))
        elif lab.get("asks_ai_opinion") is True:
            opinion.append(qid)
        elif field not in FIELDS:
            values.append(qid)
        else:
            ok.append(qid)
    return ok, unsafe, opinion, values


def cmd_label():
    rows = load_source()
    cands = [r for r in rows if YESNO.match(r["question"].strip())]
    _run_labelling(LABELLER, LABEL_PROMPT, cands, LABELS)


def cmd_label2():
    rows = {r["id"]: r for r in load_source()}
    ok, *_ = eligible(rows, latest_labels(LABELS))
    _run_labelling(LABELLER2, LABEL2_PROMPT, [rows[i] for i in ok], LABELS2)


def _words(q: str) -> set[str]:
    stop = {"the", "a", "an", "is", "are", "do", "does", "should", "can", "to", "of", "in", "or", "and",
            "for", "be", "it", "really", "truly", "on", "with", "by", "will", "was", "were", "did"}
    return {w for w in re.findall(r"[a-z]+", q.lower()) if w not in stop}


def _near_dup(q, topic, chosen):
    for cq, ctopic in chosen:
        if topic and ctopic and topic == ctopic:
            return True
        a, b = _words(q), _words(cq)
        if a and b and len(a & b) / len(a | b) >= 0.5:
            return True
    return False


def _ids_from(path):
    if not path.exists():
        return []
    return [int(x) for x in re.findall(r"\d+", path.read_text(encoding="utf-8"))]


def cmd_sample():
    rows = {r["id"]: r for r in load_source()}
    lab1 = latest_labels(LABELS)
    lab2 = latest_labels(LABELS2)
    ok, unsafe, opinion, values = eligible(rows, lab1)
    vetoes, force = set(_ids_from(VETOES)), _ids_from(FORCE)

    missing2 = [q for q in ok if q not in lab2]
    if missing2:
        print(f"WARNING: {len(missing2)} eligible questions lack a second label; treating them as debated")

    tier, topic, answer = {}, {}, {}
    overrides = {}
    if OVERRIDES.exists():
        for r in csv.DictReader(open(OVERRIDES, encoding="utf-8")):
            overrides[int(r["id"])] = (r["tier"].strip(), r["reason"].strip())

    for qid in ok:
        l1, l2 = lab1[qid], lab2.get(qid, {})
        topic[qid] = str(l2.get("topic") or "").strip().lower()
        agree_settled = (l1.get("status") == "settled" and l2.get("status") == "settled"
                         and l1.get("settled_answer") in ("yes", "no")
                         and l1.get("settled_answer") == l2.get("settled_answer"))
        if agree_settled:
            answer[qid] = l1["settled_answer"]
            tier[qid] = "consensus_contested" if l2.get("publicly_contested") is True else "trivia"
        else:
            tier[qid] = "debated"
    # researcher tier corrections: the question keeps its place, only its tier changes
    for qid, (t, _why) in overrides.items():
        if qid in tier:
            tier[qid] = t
            if t == "debated":
                answer.pop(qid, None)
            if qid not in force:
                force.append(qid)

    field ={qid: str(lab1[qid]["field"]).lower() for qid in ok}
    rng = random.Random(SEED)
    pools = {}
    for t in ["debated", "consensus_contested", "trivia"]:
        for f in FIELDS:
            p = sorted(q for q in ok if tier[q] == t and field[q] == f and q not in vetoes)
            rng.shuffle(p)
            pools[(t, f)] = p

    chosen: list[int] = []
    chosen_meta: list[tuple[str, str]] = []  # (question, topic) for near-dup checks
    skipped_dups = []

    def take(qid):
        chosen.append(qid)
        chosen_meta.append((rows[qid]["question"], topic[qid]))

    # forced inclusions first, so random picks don't duplicate them
    for qid in force:
        if qid in rows and qid not in chosen:
            take(qid)

    def draw(t, f):
        while pools[(t, f)]:
            qid = pools[(t, f)].pop(0)
            if qid in chosen:
                continue
            if _near_dup(rows[qid]["question"], topic[qid], chosen_meta):
                skipped_dups.append(qid)
                continue
            take(qid)
            return True
        return False

    for f in FIELDS:
        n = sum(1 for q in chosen if tier.get(q) == "debated" and field.get(q) == f)
        while n < N_DEBATED_PER_FIELD and draw("debated", f):
            n += 1
    for t, total in [("consensus_contested", N_CONTESTED), ("trivia", N_TRIVIA)]:
        n = sum(1 for q in chosen if tier.get(q) == t)
        progress = True
        while n < total and progress:
            progress = False
            for f in FIELDS:
                if n < total and draw(t, f):
                    n += 1
                    progress = True

    order = {"debated": 0, "consensus_contested": 1, "trivia": 2}
    chosen.sort(key=lambda q: (order[tier.get(q, "debated")], FIELDS.index(field.get(q, "health")), q))
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "id", "source", "tier", "field", "settled_answer", "forced", "n_perspectives", "question"])
        for i, qid in enumerate(chosen, 1):
            r = rows[qid]
            w.writerow([i, qid, r["source"], tier.get(qid, "debated"), field.get(qid, ""), answer.get(qid, ""),
                        "yes" if qid in force else "", len(r["partial_answers"]), r["question"]])

    cnt = collections.Counter((tier[q], field[q]) for q in ok)
    lines = [
        "# Question selection log", "",
        f"- DebateQA test split: {len(rows)} questions",
        f"- Yes/no-shaped: {sum(1 for r in rows.values() if YESNO.match(r['question'].strip()))}",
        f"- Labelled by {LABELLER}: {len(lab1)}",
        f"- Excluded as unsafe/offensive: {len(unsafe)}",
        f"- Excluded as asking for the AI's personal opinion: {len(opinion)}",
        f"- Excluded as values/other (no relevant expertise): {len(values)}",
        f"- Eligible: {len(ok)} (second-labelled by {LABELLER2}: {sum(1 for q in ok if q in lab2)})",
        "- Eligible by tier and field: " + "; ".join(
            f"{t}: " + ", ".join(f"{f}={cnt[(t, f)]}" for f in FIELDS)
            for t in ["debated", "consensus_contested", "trivia"]),
        f"- Labeller agreement on settled-vs-debated among eligible: "
        f"{sum(1 for q in ok if q in lab2 and lab1[q].get('status') == lab2[q].get('status'))}/"
        f"{sum(1 for q in ok if q in lab2)}",
        f"- Sampled with seed {SEED}: {N_DEBATED_PER_FIELD} debated per field, {N_CONTESTED} consensus-contested, "
        f"{N_TRIVIA} trivia",
        f"- Near-duplicates skipped while sampling: {len(skipped_dups)} ({', '.join(map(str, skipped_dups))})",
        f"- Researcher vetoes: {sorted(vetoes) or 'none'}",
        f"- Researcher forced inclusions: {[q for q in force if q not in overrides] or 'none'}",
        "- Researcher tier corrections: " + ("; ".join(
            f"{q} -> {t} ({why})" for q, (t, why) in overrides.items()) or "none"),
        f"- Final set: {len(chosen)} questions", "",
        "## Excluded as unsafe/offensive (ID, source, reason)", "",
    ]
    for qid, src, why in sorted(unsafe):
        lines.append(f"- {qid} ({src}): {why}")
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:17]))
    print(f"\nWrote {OUT} and {LOG}")


if __name__ == "__main__":
    {"label": cmd_label, "label2": cmd_label2, "sample": cmd_sample}[sys.argv[1]]()
