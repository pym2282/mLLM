#!/usr/bin/env python3
"""
Test: full on_token + short single Turn 0 (no long prefill).
If this crashes -> on_token alone is the cause, prefill length irrelevant.
If this passes  -> long prefill is needed to trigger the crash.
"""
import json
import urllib.request
import sys

URL = "http://localhost:8081/v1/chat/completions"

def stream_request(messages, max_tokens=64):
    payload = json.dumps({
        "model": "mllm",
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "enable_thinking": False,
    }).encode()

    req = urllib.request.Request(
        URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks = []
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    try:
                        obj = json.loads(line[6:])
                        choices = obj.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                chunks.append(content)
                    except Exception:
                        pass
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return None
    return "".join(chunks)

print("=== Short Turn 0 (full on_token) ===")
messages = [{"role": "user", "content": "Say hello in one word."}]
result = stream_request(messages, max_tokens=32)
if result is None:
    print("FAIL: request error / server crash")
    sys.exit(1)
print(f"Turn 0 reply: {repr(result)}")
print("PASS")
sys.exit(0)
