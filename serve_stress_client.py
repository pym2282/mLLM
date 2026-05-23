#!/usr/bin/env python3
"""
Multi-turn serve-mode stress client.
Sends repeated long conversations to the running mLLM server.
"""
import json
import urllib.request
import urllib.error
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

def send(messages, max_tokens=512):
    body = json.dumps({
        "model": "mllm",
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
        "enable_thinking": True,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())

messages = [{"role": "system", "content": "You are a helpful assistant."}]

for i, user_text in enumerate(TURNS):
    messages.append({"role": "user", "content": user_text})
    print(f"[Client] Turn {i}  messages={len(messages)}  "
          f"total_user_chars={sum(len(m['content']) for m in messages if m['role']=='user')}", flush=True)
    try:
        resp = send(messages, max_tokens=512)
        assistant_text = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        print(f"[Client] Turn {i} OK  "
              f"prompt_tokens={usage.get('prompt_tokens','?')}  "
              f"completion_tokens={usage.get('completion_tokens','?')}  "
              f"reply_len={len(assistant_text)}", flush=True)
        print(f"[Client] preview: {assistant_text[:120]}", flush=True)
        messages.append({"role": "assistant", "content": assistant_text})
    except Exception as e:
        print(f"[Client] Turn {i} FAILED: {e}", flush=True)
        sys.exit(1)

print("[Client] All turns completed successfully.", flush=True)
