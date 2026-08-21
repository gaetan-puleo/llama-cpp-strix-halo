#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that a DSV4 server preserves the end of a long prompt")
    parser.add_argument("--server-url", default="http://127.0.0.1:10001")
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument("--n-predict", type=int, default=8)
    args = parser.parse_args()
    if args.prompt_tokens is not None and args.prompt_tokens <= 0:
        parser.error("--prompt-tokens must be positive")

    root = Path(__file__).resolve().parent.parent
    prompt_body = "".join(
        (root / path).read_text(encoding="utf-8")
        for path in ("README.md", "CONTRIBUTING.md", "docs/build.md")
    )
    prompt_tail = "\n\nThe capital of France is"

    base_url = args.server_url.rstrip("/")
    prompt = prompt_body + prompt_tail
    if args.prompt_tokens is not None:
        def tokenize(text: str) -> list[int]:
            request = Request(
                base_url + "/tokenize",
                data=json.dumps({"content": text}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=300) as response:
                return json.load(response)["tokens"]

        body_tokens = tokenize(prompt_body)
        tail_tokens = tokenize(prompt_tail)
        body_count = args.prompt_tokens - len(tail_tokens)
        if body_count < 0:
            parser.error("--prompt-tokens is shorter than the required prompt tail")
        repeats = (body_count + len(body_tokens) - 1) // len(body_tokens)
        prompt = (body_tokens * repeats)[:body_count] + tail_tokens

    body = {
        "prompt": prompt,
        "temperature": 0,
        "n_predict": args.n_predict,
        "cache_prompt": False,
        "stream": False,
    }

    request = Request(
        base_url + "/completion",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=300) as response:
            result = json.load(response)
    except (HTTPError, URLError, TimeoutError) as error:
        print(f"request failed: {error}")
        return 1

    evaluated = result.get("tokens_evaluated", 0)
    content = result.get("content", "")
    expected = args.prompt_tokens or 10000
    if evaluated < expected:
        print(f"prompt was unexpectedly short: {evaluated} tokens")
        return 1
    if not content.lstrip().startswith("Paris."):
        print(f"long-context completion was corrupted: {content!r}")
        return 1

    stats = {
        key: value
        for key, value in result.items()
        if "accept" in key or "draft" in key or "timing" in key
    }
    print(f"long-context completion passed: {evaluated} prompt tokens")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
