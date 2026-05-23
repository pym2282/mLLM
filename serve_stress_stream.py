#!/usr/bin/env python3
"""
Multi-turn STREAMING stress client for mLLM serve mode.
Sends repeated long conversations using SSE streaming.
"""
import json
import urllib.request
import sys

URL = "http://localhost:8081/v1/chat/completions"

TURNS = [
    "Please explain in detail what machine learning is. Include key concepts, "
    "types of learning, and real-world applications. Be thorough.",
    "Now explain deep learning and how it differs from classical ML. "
    "Describe neural networks, layers, and backpropagation in detail.",
    "Describe transformer architecture thoroughly. Explain self-attention, "
    "multi-head attention, positional encoding, and why transformers dominate NLP.",
    "Explain how large language models like GPT are trained. Describe "
    "pretraining objectives, fine-tuning, RLHF, and inference at scale.",
    "Summarize all four topics above in a concise paragraph each. "
    "Then discuss future directions for AI research.",
]

def send_stream(messages, max_tokens=512):
    """Send streaming request and return the full assistant reply text."""
    body = json.dumps({
        "model": "mllm",
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "enable_thinking": True,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                  headers={"Content-Type": "application/json"})

    full_text = ""
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw_line in r:
            line = raw_line.decode("utf-8").rstrip("\n\r")
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                full_text += content
    return full_text

messages = [{"role": "system", "content": "You are a helpful assistant."}]

for i, user_text in enumerate(TURNS):
    messages.append({"role": "user", "content": user_text})
    total_chars = sum(len(m["content"]) for m in messages if m.get("role") == "user")
    print(f"[Client] Turn {i}  messages={len(messages)}  total_user_chars={total_chars}", flush=True)

    try:
        assistant_text = send_stream(messages, max_tokens=512)
        print(f"[Client] Turn {i} OK  reply_len={len(assistant_text)}", flush=True)
        print(f"[Client] preview: {assistant_text[:120]}", flush=True)
        messages.append({"role": "assistant", "content": assistant_text})
    except Exception as e:
        print(f"[Client] Turn {i} FAILED: {e}", flush=True)
        sys.exit(1)

print("[Client] All turns completed successfully.", flush=True)
