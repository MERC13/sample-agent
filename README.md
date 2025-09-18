# Sample A2A Agent (Flask + Ollama)

A minimal A2A-compliant JSON-RPC agent with:
- LLM chat (Ollama)
- Image understanding (**must only return text, json format is rejected**)
- Hash tool sequencing (md5/sha1/sha256/sha384/sha512)
- Prime-sum shortcut (mod 1000)
- Simple persistent number memory
- Safe code generation and execution
- Web browsing task shortcut

## Features

- JSON-RPC 2.0 endpoint for A2A clients
- Agent card served at [/.well-known/agent-card.json](.well-known/agent-card.json) (also written at runtime by the route).
- Persistent memory stored in [memory.json](memory.json).

## Requirements

- Python 3.10+
- Ollama running locally (for LLM and vision features)

Environment variables:
- OLLAMA_MODEL (default: llava)
- CHAT_MEMORY_FILE (default: memory.json)

## Install (Windows)

```sh
python -m venv .venv
. .venv/Scripts/activate

pip install -r requirements.txt
ollama pull llava
```

## Run

```sh
python app.py
```

CORS is limited to the origin https://ape.llm.phd.

## API

- GET /.well-known/agent-card.json — serves the agent card (also persists it).
  - Route: agent_card in [app.py](app.py)
- POST / or /rpc — JSON-RPC 2.0
  - method: "message/send"
  - params.message.parts: array of A2A parts (supports "text"; image bytes can be passed via params.bytes or a file part’s file.bytes)

## Implementation Notes

- Memory file: [memory.json](memory.json) (configurable via CHAT_MEMORY_FILE).
- Safe execution:
  - Allowed imports: math, statistics, itertools
  - Safe builtins only; AST validation in [`validate_ast`](app.py) and executed by [`safe_execute`](app.py).
- If Ollama is not available, chat returns "Ollama model not available." and code generation is skipped.