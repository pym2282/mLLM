#!/usr/bin/env python3
"""
Test: Full on_token + SHORT prefill + max_tokens=950
Goal: force kv_seq >= 938 with a short initial prompt.
If this PASSES  -> prefill length itself is the variable (not kv_seq threshold).
If this CRASHES -> kv_seq ~938 threshold is the cause regardless of prefill length.
"""
import json
import urllib.request
import sys

URL = "http://localhost:8081/v1/chat/completions"

def stream_request(messages, max_tokens=64, enable_thinking=False):
    payload = json.dumps({
        "model": "mllm",
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
    }).encode()

    req = urllib.request.Request(
        URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks = []
    tok_count = 0
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
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
                                tok_count += 1
                                if tok_count % 100 == 0:
                                    print(f"  ... received {tok_count} chunks so far", flush=True)
                    except Exception:
                        pass
    except Exception as e:
        print(f"[ERROR after {tok_count} chunks] {e}", file=sys.stderr)
        return None, tok_count
    return "".join(chunks), tok_count

print("=== Short prefill + max_tokens=950 (kv_seq target ~994) ===")
print("Prompt: 'Say hello in one word.'  (prompt ~10 tokens)")
messages = [{"role": "user", "content": "Say hello in one word."}]
result, n = stream_request(messages, max_tokens=950, enable_thinking=False)
if result is None:
    print(f"FAIL: server crashed after {n} decode steps")
    sys.exit(1)
print(f"PASS: received {n} chunks, total reply len={len(result)}")
print(f"Reply preview: {repr(result[:200])}")
sys.exit(0)
