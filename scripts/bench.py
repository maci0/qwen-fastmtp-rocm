#!/usr/bin/env python3
"""Benchmark streaming decode speed and latency for Qwen3.8 FastMTP endpoint."""

import argparse
import json
import time
import urllib.request


def run_benchmark(endpoint: str, prompt: str, max_tokens: int, temperature: float) -> None:
    payload = {
        "model": "qwen",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
    }

    req = urllib.request.Request(
        f"{endpoint}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    start_time = time.time()
    first_token_time = None
    token_count = 0
    chunks: list[str] = []

    with urllib.request.urlopen(req) as resp:
        for line in resp:
            raw = line.decode("utf-8").strip()
            if not raw or not raw.startswith("data: "):
                continue
            data = raw[6:]
            if data == "[DONE]":
                break
            try:
                msg = json.loads(data)
                delta = msg["choices"][0].get("delta", {})
                content = delta.get("content") or delta.get("reasoning_content")
                if content:
                    if first_token_time is None:
                        first_token_time = time.time()
                    token_count += 1
                    chunks.append(content)
            except Exception:
                pass

    end_time = time.time()
    ttft = (first_token_time - start_time) if first_token_time else 0.0
    duration = (end_time - first_token_time) if first_token_time else 0.0
    tok_per_sec = (token_count / duration) if duration > 0 else 0.0

    print(f"Generated: {token_count} tokens")
    print(f"TTFT:      {ttft:.3f} s")
    print(f"Duration:  {duration:.3f} s")
    print(f"Speed:     {tok_per_sec:.2f} tokens/s")
    print("--- Sample output ---")
    print("".join(chunks)[:240].strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastMTP benchmark client")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080", help="llama-server endpoint URL")
    parser.add_argument("--max-tokens", type=int, default=512, help="maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="sampling temperature")
    parser.add_argument(
        "--prompt",
        default="Write a thread-safe ring buffer in C with pthread mutex and condition variables.",
        help="input prompt",
    )
    args = parser.parse_args()
    run_benchmark(args.endpoint, args.prompt, args.max_tokens, args.temperature)
