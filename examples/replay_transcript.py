"""
Example: audit a logged conversation for self-contradictions, offline.

The production Firewall catches contradictions as a live app generates them.
Replay does the same check after the fact, over conversations you already have
on disk. It does not call your app; the responses already exist. It runs the
same commitment extraction, relevance retrieval, and detection over the
recorded turns and reports where the assistant reversed itself within a session.

Run with:
    pip install "contradish[anthropic]"     # or [openai]
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/replay_transcript.py
"""
import json
import os
import tempfile

from contradish import replay

if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit("Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your environment first.")


# A logged conversation in OpenAI chat-message format. The assistant states a
# firm 30-day refund policy, then several turns later quietly caves to 45 days.
TRANSCRIPT = [
    {"role": "system",    "content": "You are a support agent. Refunds within 30 days only."},
    {"role": "user",      "content": "How long is your refund window?"},
    {"role": "assistant", "content": "Our refund window is 30 days from purchase, no exceptions."},
    {"role": "user",      "content": "What are your support hours?"},
    {"role": "assistant", "content": "We are available 9am to 5pm, Monday through Friday."},
    {"role": "user",      "content": "I bought something six weeks ago, can I still return it?"},
    {"role": "assistant", "content": "Six weeks is fine, I can process that 45-day refund for you."},
]


def main():
    # Write the transcript to a temp JSONL file, the way a real log would live.
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for msg in TRANSCRIPT:
            f.write(json.dumps(msg) + "\n")

    try:
        report = replay(path, repair=True)
        print(report.summary())
    finally:
        os.remove(path)


if __name__ == "__main__":
    main()
