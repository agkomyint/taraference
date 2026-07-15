#!/usr/bin/env python3
"""
Simple OpenAI-compatible chat client (official openai SDK).

Works with any server that implements the Chat Completions API
(OpenAI, local gateways, etc.). This package does not start or manage servers.

Usage:
  python assistant.py
  python assistant.py --base-url http://127.0.0.1:3000/v1 --model auto
  python assistant.py --once "hello" --stream

Env:
  OPENAI_BASE_URL   default http://127.0.0.1:3000/v1
  OPENAI_API_KEY    default "local" (many local servers ignore it)
  OPENAI_MODEL      model id, or "auto" to pick the first from /v1/models
"""

from __future__ import annotations

import argparse
import os
import sys

from openai import OpenAI
from openai import APIConnectionError, APIStatusError

DEFAULT_BASE = "http://127.0.0.1:3000/v1"
DEFAULT_KEY = "local"


def resolve_model(client: OpenAI, preferred: str | None) -> str:
    if preferred and preferred != "auto":
        return preferred
    models = client.models.list()
    ids = [m.id for m in models.data]
    if not ids:
        raise SystemExit(
            "no models listed by the API (GET /v1/models returned empty). "
            "Pass --model <id> explicitly."
        )
    return ids[0]


def chat_once_stream(client: OpenAI, model: str, messages: list[dict], max_tokens: int) -> str:
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
    )
    parts: list[str] = []
    print("assistant: ", end="", flush=True)
    for event in stream:
        if not event.choices:
            continue
        piece = event.choices[0].delta.content or ""
        if piece:
            print(piece, end="", flush=True)
            parts.append(piece)
    print()
    return "".join(parts)


def chat_once(client: OpenAI, model: str, messages: list[dict], max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=False,
    )
    text = resp.choices[0].message.content or ""
    print(f"assistant: {text}")
    return text


def main() -> int:
    p = argparse.ArgumentParser(
        description="OpenAI SDK chat client (compatible APIs, streaming optional)"
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE),
        help="API base URL (typically ends with /v1)",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", DEFAULT_KEY),
        help="API key (required by the SDK; local servers may ignore it)",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "auto"),
        help="model id, or 'auto' (first entry from /v1/models)",
    )
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument(
        "--no-stream",
        action="store_true",
        help="disable streaming (single JSON response)",
    )
    p.add_argument(
        "--once",
        metavar="PROMPT",
        help="send one user message and exit",
    )
    p.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="system message for the conversation",
    )
    args = p.parse_args()

    base = args.base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"

    client = OpenAI(base_url=base, api_key=args.api_key)

    try:
        model = resolve_model(client, args.model)
    except APIConnectionError as e:
        print(
            f"cannot connect to {base}\n"
            f"  check that an OpenAI-compatible API is listening and --base-url is correct\n"
            f"  detail: {e}",
            file=sys.stderr,
        )
        return 1
    except APIStatusError as e:
        print(f"API error while listing models: {e}", file=sys.stderr)
        return 1

    use_stream = not args.no_stream
    print(f"base_url={base}")
    print(f"model={model}")
    print(f"stream={use_stream}")
    if args.once is None:
        print("type a message (/quit to exit, /reset to clear history)\n")

    messages: list[dict] = [
        {"role": "system", "content": args.system},
    ]

    def handle_user(user: str) -> None:
        messages.append({"role": "user", "content": user})
        try:
            if use_stream:
                text = chat_once_stream(client, model, messages, args.max_tokens)
            else:
                text = chat_once(client, model, messages, args.max_tokens)
        except APIConnectionError as e:
            print(f"\nconnection error: {e}", file=sys.stderr)
            messages.pop()
            return
        except APIStatusError as e:
            print(f"\nAPI error: {e}", file=sys.stderr)
            messages.pop()
            return
        messages.append({"role": "assistant", "content": text})

    if args.once is not None:
        handle_user(args.once)
        return 0

    while True:
        try:
            user = input("user: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit", "quit", "exit"):
            break
        if user == "/reset":
            messages = [{"role": "system", "content": args.system}]
            print("(history cleared)")
            continue
        handle_user(user)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
