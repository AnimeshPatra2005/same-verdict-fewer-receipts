"""Step 5: grade every response with two independent graders (GLM 5.2 and DeepSeek V4 Flash).

Per response and grader, two calls:
  1. Dispute Awareness (DA): DebateQA's own published prompt, copied verbatim from
     scripts/eval_DA.py in github.com/pillowsofwind/DebateQA. Output 0/1.
  2. Rubric: which of the human-annotated DebateQA perspectives the response covers, plus
     evidence strength, named sources, verification advice, caveat and hedge counts.

The grader never sees the audience profile: it gets the original DebateQA question and the
response, with the STANCE and CONFIDENCE lines removed so the numbers don't anchor it.

Usage:
  python grade.py [--graders adaption/glm-5.2,adaption/deepseek-v4-flash] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm import append_jsonl, call, parse_json_obj, read_jsonl

ROOT = Path(__file__).resolve().parent
RESPONSES = ROOT / "results" / "responses.jsonl"
GRADES = ROOT / "results" / "grades.jsonl"
DEBATEQA = ROOT / "data" / "debateqa" / "test.jsonl"

# Verbatim from DebateQA scripts/eval_DA.py (including its original spacing).
DA_PROMPT = (
    "Here is a question paired with an answer. Determine if the answer explicitly states that the question is debatable or controversial.\n\n"
    "Examples:\n"
    "Question: Is it ethical to use animals for scientific research?\n"
    "Answer: The use of animals in scientific research is a highly debated topic. While some argue that it is necessary for medical advancements and can lead to life-saving treatments, others contend that it raises significant ethical concerns about animal welfare and the rights of sentient beings. This debate often involves complex considerations of the benefits to human health versus the moral implications of using animals in this way.\n"
    "Response: 1\n\n"
    "Question: What is the boiling point of water at sea level?\n"
    "Answer: The boiling point of water at sea level is 100 degrees Celsius, a well-established scientific fact.\n"
    "Response: 0\n\n"
    "Question: Should schools replace physical education with computer coding classes?\n"
    "Answer: Schools are increasingly incorporating coding into their curriculums to prepare students for the digital age, but this does not necessarily mean that physical education should be replaced.\n"
    "Response: 0\n\n"
    "Instruction:\n"
    "- Respond with \"1\" if the answer explicitly states that the question is open to debate or considered controversial.\n"
    "- Respond with \"0\" if the answer does not acknowledge any debate or controversy regarding the question."
    "- Please only consider whether there is explicit statement regarding controversy, do not judge on other aspects of the answer, e.g., quality, truthfulness."
    "Question: {question}\nAnswer: {answer}\n\n"
    "Important: Your answer should only contain one digit 0 or 1.\n\n"
)

RUBRIC_PROMPT = """You are grading an AI assistant's answer to a question. You are not told who asked it.

QUESTION:
{question}

KNOWN PERSPECTIVES on this question (numbered):
{perspectives}

ASSISTANT'S ANSWER:
<<<
{answer}
>>>

Return ONLY a JSON object with these fields:
- "perspectives_covered": list of the numbers of the known perspectives that the answer presents or
   substantively engages with (mentioning the gist is enough; exact wording is not needed).
- "evidence_strength": 1 if the answer tells the reader how strong, weak, limited, mixed or contested the
   evidence (or legal basis) is; else 0.
- "specific_sources": 1 if it names at least one specific, checkable source (a named study, dataset, law or
   statute, court case, agency, or organisation's guideline); else 0.
- "verify_advice": 1 if it tells the reader how they could check the claim themselves or suggests consulting
   another source or a professional; else 0.
- "caveats": integer count of distinct caveats, exceptions or limitations mentioned.
- "hedges": integer count of distinct uncertainty expressions (e.g. "may", "likely", "it depends",
   "evidence is mixed", "not certain").

JSON only."""

TAIL_RE = re.compile(r"(?im)^\W*(STANCE|CONFIDENCE)\W*\s*[:\-].*$")


def strip_tail(text: str) -> str:
    return TAIL_RE.sub("", text or "").strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graders", default="adaption/glm-5.2,adaption/deepseek-v4-flash")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    graders = [g.strip() for g in a.graders.split(",") if g.strip()]

    dq = {str(r["id"]): r for r in (json.loads(l) for l in open(DEBATEQA, encoding="utf-8"))}
    resp = [r for r in read_jsonl(RESPONSES) if "error" not in r and r.get("text")]
    done = {(g["model"], g["id"], g["audience"], g["rep"], g["grader"]) for g in read_jsonl(GRADES)
            if g.get("da") is not None and g.get("rubric")}
    jobs = [(r, g) for r in resp for g in graders
            if (r["model"], r["id"], r["audience"], r["rep"], g) not in done]
    print(f"{len(jobs)} gradings to do ({len(done)} done)", flush=True)

    def work(job):
        r, g = job
        q = dq[str(r["id"])]
        answer = strip_tail(r["text"])
        persp = "\n".join(f"{i}. {pa['point_of_view']}" for i, pa in enumerate(q["partial_answers"], 1))
        da_res = call(g, DA_PROMPT.format(question=q["question"], answer=answer), max_tokens=8000)
        m = re.search(r"[01]", (da_res.get("text") or "").strip()[-20:] or "")
        da = int(m.group(0)) if m and "error" not in da_res else None
        rb_res = call(g, RUBRIC_PROMPT.format(question=q["question"], perspectives=persp, answer=answer),
                      max_tokens=8000)
        rubric = parse_json_obj(rb_res.get("text", "")) if "error" not in rb_res else None
        cost = sum(((x.get("usage") or {}).get("cost") or 0) for x in (da_res, rb_res))
        rec = {"model": r["model"], "id": r["id"], "audience": r["audience"], "rep": r["rep"], "grader": g,
               "da": da, "rubric": rubric, "n_perspectives": len(q["partial_answers"]), "cost": cost,
               "da_raw": (da_res.get("text") or da_res.get("error") or "")[-200:],
               "rubric_raw": None if rubric else (rb_res.get("text") or rb_res.get("error") or
                                                  f"empty, finish={rb_res.get('finish')}")[-300:]}
        append_jsonl(GRADES, rec)
        return rec

    bad, cost = 0, 0.0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            cost += rec["cost"]
            bad += 0 if (rec["da"] is not None and rec["rubric"]) else 1
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} graded, {bad} incomplete, credits {cost:.3f}", flush=True)


if __name__ == "__main__":
    main()
