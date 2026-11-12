"""Model-backed fact extraction for InfinityDB — optional, no dependencies.

The engine is deterministic by design: it stores what was said, verbatim, with
provenance. That is a floor, not a ceiling. ``add(..., extractor=fn)`` accepts
any callable that turns a conversation into facts, and this module supplies the
two common ones so callers do not have to write the glue:

    from infinitydb import InfinityDB
    from infinitydb.extraction import openai_extractor, ollama_extractor

    db = InfinityDB("memory.db")
    db.add(turn, user_id="u1", extractor=openai_extractor())          # hosted
    db.add(turn, user_id="u1", extractor=ollama_extractor())          # local, free

Everything here uses ``urllib`` from the standard library: the core engine keeps
zero third-party dependencies, and an extraction call is the only thing that
leaves the machine. Nothing is extracted unless you pass an extractor.

The extractor is called with the raw turn text and must return facts shaped as
``{"fact": str, "category": str, "importance": float}``; the engine validates
those fields and raises if the model returns something else, so a broken model
response can never silently store garbage.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Callable

__all__ = ["chat_extractor", "openai_extractor", "ollama_extractor", "parse_facts", "main"]

SYSTEM_PROMPT = """You extract long-term memory from a single conversation turn.

Rules:
- Only durable, self-contained facts about the people and things mentioned.
- One fact per statement, in the third person, no pronouns ("Caroline works at
  a veterinary clinic", not "I work here").
- Ignore pleasantries, filler, and anything the speaker does not assert.
- category is one of: fact, preference, dislike, event, profile, relationship.
- importance is 0.0-1.0; use 0.9+ only for stable, identity-level facts.
- Never invent. If the turn contains no durable fact, return an empty list.

Reply with JSON only: {"facts": [{"fact": "...", "category": "fact", "importance": 0.8}]}"""

_USER_TEMPLATE = "Conversation turn:\n{turn}"


def parse_facts(raw: str, max_facts: int = 20) -> list[dict[str, Any]]:
    """Parse a model reply into the dicts ``add(..., extractor=...)`` accepts.

    Tolerates the usual model habits: a fenced code block, a bare JSON array,
    or an object wrapping the array under ``facts``/``memories``/``items``.
    Raises ``ValueError`` rather than returning ``[]``: an unparsable reply is
    a failure, and silent emptiness is how memory systems lose facts.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("extractor reply was empty")
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = min((i for i in (text.find("["), text.find("{")) if i >= 0), default=-1)
        if start < 0:
            raise ValueError("extractor reply contained no JSON") from None
        try:
            payload, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as exc:
            raise ValueError(f"extractor reply was not valid JSON: {exc}") from exc

    if isinstance(payload, dict):
        for key in ("facts", "memories", "items", "results"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("extractor reply must be a JSON list or an object with a facts list")

    facts: list[dict[str, Any]] = []
    for item in payload[:max_facts]:
        if isinstance(item, str):
            item = {"fact": item}
        if not isinstance(item, dict):
            continue
        fact = str(item.get("fact") or item.get("text") or "").strip()
        if not fact:
            continue
        category = str(item.get("category") or "fact").strip().lower()
        if not re.fullmatch(r"[a-z_]{1,32}", category):
            category = "fact"
        try:
            importance = float(item.get("importance", 0.8))
        except (TypeError, ValueError):
            importance = 0.8
        facts.append({
            "fact": fact[:2000],
            "category": category,
            "importance": min(1.0, max(0.0, importance)),
        })
    return facts


def chat_extractor(
    *,
    base_url: str,
    model: str,
    api_key: str | None = None,
    api_key_env: str | None = None,
    timeout: float = 60.0,
    max_facts: int = 20,
    temperature: float = 0.0,
) -> Callable[[str], list[dict[str, Any]]]:
    """Build an extractor that calls any OpenAI-compatible ``/chat/completions``.

    Works with hosted APIs and with local servers (Ollama, llama.cpp, vLLM).
    The key is read from ``api_key`` or from the named environment variable, and
    is only ever sent in the request header — never logged or stored.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url cannot be empty")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model cannot be empty")
    url = base_url.rstrip("/") + "/chat/completions"

    def extract(turn: str) -> list[dict[str, Any]]:
        key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        body = json.dumps({
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(turn=turn)},
            ],
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(
                f"extraction request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"extraction endpoint unreachable: {exc.reason}") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("extraction reply had no message content") from exc
        return parse_facts(content, max_facts=max_facts)

    return extract


def openai_extractor(
    model: str = "gpt-4o-mini",
    *,
    api_key: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: str = "https://api.openai.com/v1",
    **kwargs: Any,
) -> Callable[[str], list[dict[str, Any]]]:
    """Hosted OpenAI-compatible extractor. Needs a key; raises if it is missing."""
    if not api_key and not os.environ.get(api_key_env):
        raise RuntimeError(
            f"{api_key_env} is not set: pass api_key=..., set the variable, or use "
            "ollama_extractor() for a local model"
        )
    return chat_extractor(
        base_url=base_url, model=model, api_key=api_key, api_key_env=api_key_env, **kwargs
    )


def ollama_extractor(
    model: str = "llama3.1:8b",
    *,
    base_url: str = "http://127.0.0.1:11434/v1",
    **kwargs: Any,
) -> Callable[[str], list[dict[str, Any]]]:
    """Local, free extractor for an Ollama (or compatible) server — no key."""
    return chat_extractor(base_url=base_url, model=model, **kwargs)


SAMPLE_TURN = (
    "Caroline: I work at a veterinary clinic in Bristol, and I moved here in May."
)


def main(argv: list[str] | None = None) -> int:
    """Verify a backend end to end: ``python -m infinitydb.extraction --check``.

    This is the one command to run the day a key lands. It makes the same call
    the engine makes, prints the parsed facts, and exits non-zero with the real
    error when the key, base URL or model is wrong — instead of the failure only
    appearing later, mid-ingest.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m infinitydb.extraction",
        description="Check an OpenAI-compatible extraction backend before using it.",
    )
    parser.add_argument("--check", action="store_true", help="run one sample extraction")
    parser.add_argument("--backend", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--text", default=SAMPLE_TURN)
    args = parser.parse_args(argv)

    if not args.check:
        parser.print_help()
        return 2
    try:
        if args.backend == "ollama":
            extractor = ollama_extractor(
                model=args.model or "llama3.1:8b",
                **({"base_url": args.base_url} if args.base_url else {}),
            )
        else:
            extractor = openai_extractor(
                model=args.model or "gpt-4o-mini",
                **({"base_url": args.base_url} if args.base_url else {}),
            )
        facts = extractor(args.text)
    except (RuntimeError, ValueError) as exc:
        print(f"extraction check FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"facts": facts}, indent=2))
    print(f"OK: {len(facts)} fact(s) extracted from the sample turn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
