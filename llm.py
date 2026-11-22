"""Thin client for the two model providers used in this project.

Model specs are written "provider/model-id", e.g.
    coarena/claude-sonnet-5-5     (OpenAI-compatible chat completions)
    adaption/glm-5.2              (OpenAI Responses API)

Keys are read from the environment or from a .env file next to this script
(COARENA_API_KEY, ADAPTION_API_KEY). Keys are never printed or logged.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent

ENDPOINTS = {
    "coarena": "https://coarena.ai/api/v1/chat/completions",
    "adaption": "https://api.prod.adaptionlabs.ai/api/v1/responses",
}
KEY_NAMES = {"coarena": "COARENA_API_KEY", "adaption": "ADAPTION_API_KEY"}


class QuotaExhausted(RuntimeError):
    """Raised when a provider's daily budget is used up; retrying won't help today."""


_keys: dict[str, str] = {}


def _key(provider: str) -> str:
    if provider in _keys:
        return _keys[provider]
    name = KEY_NAMES[provider]
    val = os.environ.get(name)
    if not val:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8-sig").splitlines():
                if line.strip().startswith(name + "="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not val:
        raise SystemExit(f"Missing {name} (set it in .env)")
    _keys[provider] = val
    return val


# Coarena allows 30 requests/min per account: space requests out globally.
_pace_lock = threading.Lock()
_last_call = {"coarena": 0.0, "adaption": 0.0}
_MIN_GAP = {"coarena": 2.1, "adaption": 0.0}


def _pace(provider: str) -> None:
    gap = _MIN_GAP[provider]
    if not gap:
        return
    with _pace_lock:
        wait = _last_call[provider] + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call[provider] = time.time()


def _extract_responses_text(data: dict) -> str:
    if isinstance(data.get("output_text"), str) and data["output_text"]:
        return data["output_text"]
    chunks = []
    for item in data.get("output", []) or []:
        if item.get("type") == "message":
            for c in item.get("content", []) or []:
                if c.get("type") in ("output_text", "text") and c.get("text"):
                    chunks.append(c["text"])
    return "\n".join(chunks)


def call(spec: str, prompt: str, max_tokens: int = 4000, retries: int = 6) -> dict:
    """Call a model once. Returns a dict with text, served_model, usage, request_id, latency_s,
    or a dict with 'error'. Raises QuotaExhausted when the daily budget is gone."""
    provider, model = spec.split("/", 1)
    url = ENDPOINTS[provider]
    headers = {"Authorization": f"Bearer {_key(provider)}", "Content-Type": "application/json"}
    if provider == "coarena":
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    else:
        body = {"model": model, "input": prompt, "max_output_tokens": max_tokens}

    delay = 5.0
    last_err = None
    for attempt in range(retries):
        _pace(provider)
        t0 = time.time()
        try:
            r = requests.post(url, headers=headers, json=body, timeout=300)
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(delay + random.random())
            delay = min(delay * 2, 90)
            continue
        latency = time.time() - t0
        if r.status_code == 200:
            data = r.json()
            if provider == "coarena":
                choice = (data.get("choices") or [{}])[0]
                text = (choice.get("message") or {}).get("content") or ""
                served = r.headers.get("x-coarena-model") or data.get("model")
                req_id = r.headers.get("x-coarena-request-id") or data.get("id")
                finish = choice.get("finish_reason")
            else:
                text = _extract_responses_text(data)
                served = data.get("model")
                req_id = data.get("id")
                finish = data.get("status")
            return {"text": text, "served_model": served, "usage": data.get("usage"),
                    "request_id": req_id, "finish": finish, "latency_s": round(latency, 2),
                    "attempts": attempt + 1}
        # errors
        snippet = r.text[:400]
        if r.status_code == 429:
            if r.headers.get("x-should-retry") == "false" or "insufficient_quota" in snippet:
                raise QuotaExhausted(f"{provider}: daily quota reached")
            ra = r.headers.get("retry-after")
            time.sleep(float(ra) if ra and ra.replace(".", "", 1).isdigit() else delay)
            delay = min(delay * 2, 90)
            last_err = f"HTTP 429: {snippet}"
            continue
        if r.status_code in (400, 401, 403, 404, 422):
            return {"error": f"HTTP {r.status_code}: {snippet}"}
        last_err = f"HTTP {r.status_code}: {snippet}"
        time.sleep(delay + random.random())
        delay = min(delay * 2, 90)
    return {"error": last_err or "unknown error"}


_io_lock = threading.Lock()


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _io_lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def parse_json_obj(text: str) -> dict | None:
    import re
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    import sys
    spec = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Reply with just the word OK."
    res = call(spec, prompt, max_tokens=int(sys.argv[3]) if len(sys.argv) > 3 else 400)
    print(json.dumps({k: v for k, v in res.items()}, indent=2, ensure_ascii=False))
