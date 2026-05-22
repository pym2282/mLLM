#!/usr/bin/env python3
"""
Test: long prefill with no-op on_token.
Forces a ~600-token assistant reply into Turn 1 context to replicate
the prefill length from the original crash (kv_seq~676 before decode).
"""
import json
import urllib.request
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = "http://localhost:8081/v1/chat/completions"

# ~600 tokens of real prose to simulate a long Turn 0 assistant reply
FAKE_ASSISTANT_REPLY = (
    "Machine learning is a subfield of artificial intelligence that focuses on "
    "developing algorithms and statistical models that enable computer systems to "
    "learn and improve from experience without being explicitly programmed. At its "
    "core, machine learning involves training a model on a dataset, allowing it to "
    "identify patterns and relationships, and then using those patterns to make "
    "predictions or decisions on new, unseen data. The field encompasses several "
    "key concepts including supervised learning, where models are trained on labeled "
    "data with known outputs; unsupervised learning, where algorithms discover hidden "
    "patterns in unlabeled data; and reinforcement learning, where agents learn to "
    "take actions in an environment to maximize cumulative rewards. Deep learning, a "
    "subset of machine learning, uses neural networks with many layers to automatically "
    "learn hierarchical representations of data, achieving remarkable success in image "
    "recognition, natural language processing, speech recognition, and game playing. "
    "Real-world applications of machine learning span numerous industries: healthcare "
    "uses ML for disease diagnosis and drug discovery; finance employs it for fraud "
    "detection and algorithmic trading; autonomous vehicles rely on it for perception "
    "and decision-making; recommendation systems on streaming platforms and e-commerce "
    "sites leverage collaborative filtering and content-based approaches to personalize "
    "user experiences. The mathematical foundations include linear algebra, calculus, "
    "probability theory, and statistics, while practical implementation requires "
    "proficiency in frameworks such as TensorFlow, PyTorch, and scikit-learn. As "
    "datasets grow larger and computational resources become more powerful through "
    "GPUs and specialized accelerators, machine learning models continue to scale "
    "in complexity and capability, pushing the boundaries of what artificial "
    "intelligence can achieve in perception, reasoning, and generation tasks. "
    "Supervised learning algorithms such as linear regression, logistic regression, "
    "support vector machines, decision trees, and random forests form the backbone "
    "of classical machine learning. These methods require labeled training examples "
    "where each input is paired with the correct output, enabling the model to learn "
    "a mapping function from inputs to outputs. Gradient descent optimization iteratively "
    "adjusts model parameters to minimize a loss function that quantifies prediction "
    "error. Regularization techniques such as L1 and L2 penalties prevent overfitting "
    "by constraining parameter magnitudes, while cross-validation provides unbiased "
    "estimates of generalization performance on held-out data. Unsupervised learning "
    "encompasses clustering algorithms like k-means and DBSCAN that partition data "
    "into groups based on similarity, as well as dimensionality reduction methods "
    "such as principal component analysis and t-SNE that project high-dimensional "
    "data onto lower-dimensional manifolds for visualization and feature extraction. "
    "Generative models including variational autoencoders and generative adversarial "
    "networks learn to synthesize new data samples that match the statistical "
    "distribution of training data, enabling applications in image synthesis, "
    "data augmentation, and anomaly detection. The bias-variance tradeoff describes "
    "a fundamental tension in model selection: simple models exhibit high bias and "
    "low variance, underfitting the data, while complex models exhibit low bias and "
    "high variance, overfitting to noise in the training set. Ensemble methods such "
    "as bagging, boosting, and stacking combine multiple weak learners to produce "
    "a stronger predictor with improved generalization. Feature engineering, the "
    "process of transforming raw data into informative representations, remains "
    "critical even in the era of deep learning, particularly for tabular and "
    "structured data domains where domain expertise can encode valuable prior knowledge."
)

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
        print(f"[Client] preview: {assistant_text[:80]}", flush=True)

        # After Turn 0: override reply with long fake text to force large prefill in Turn 1
        if i == 0:
            print(f"[Client] Injecting long fake reply for Turn 1 prefill test", flush=True)
            messages.append({"role": "assistant", "content": FAKE_ASSISTANT_REPLY})
        else:
            messages.append({"role": "assistant", "content": assistant_text})
    except Exception as e:
        print(f"[Client] Turn {i} FAILED: {e}", flush=True)
        sys.exit(1)

print("[Client] All turns completed successfully.", flush=True)
