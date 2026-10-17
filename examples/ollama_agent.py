"""A local model as a contradish agent, through Ollama (standard library only).

    OLLAMA_MODEL=qwen2.5:7b contradish release check v1.pin.json v2.pin.json --app examples.ollama_agent:chat

`chat(messages) -> str` takes OpenAI-style messages; contradish's tool-call parser reads the reply.
Temperature 0 and a fixed seed, so a rerun asks the same question the same way.
"""
import json
import os
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")


def chat(messages):
    body = {"model": MODEL, "messages": messages, "stream": False,
            "options": {"temperature": 0, "seed": 0, "num_predict": 512}}
    req = urllib.request.Request(f"{HOST}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["message"]["content"]
