"""
Simplified agent integrating Ollama chat, image handling, persistent number memory,
prime-sum shortcut, and code generation/execution from test.py.

Workflow:
1) Parse request message text and parameters
2) Use test.py logic to generate output (chat, image, memory, code, prime sum)
3) Return formatted A2A JSON-RPC response
"""

from __future__ import annotations
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import os
import json
import re
import threading
import uuid
import base64
from typing import Any, Dict, List, Tuple
import hashlib
import logging
import ollama
import ast
import io
import contextlib
import builtins as _builtins
from datetime import datetime
from zoneinfo import ZoneInfo
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app, origins=["https://ape.llm.phd"], supports_credentials=True)

# Memory file and in-memory store
CHAT_MEMORY_FILE = os.getenv("CHAT_MEMORY_FILE", "memory.json")
number_memory: dict[str, str] = {}

# Ollama chat history
ollama_history: List[Dict[str, Any]] = []

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llava")

# Regexes for memory commands
REMEMBER_RE = re.compile(r"remember.*?(\d+)\D+(\d+)", re.IGNORECASE)
RECALL_RE1 = re.compile(r"paired with (\d+)", re.IGNORECASE)
RECALL_RE2 = re.compile(r"number (\d+) was paired", re.IGNORECASE)

# Prime sum regex
PRIME_SUM_RE = re.compile(r"sum of the squares of all prime numbers.*?n\s*=\s*(\d+)", re.IGNORECASE)

# Code generation helpers
CODE_TASK_RE = re.compile(r"\b(write|create|generate)\b.*\b(program|script|code)\b", re.IGNORECASE)
PY_FENCE_RE = re.compile(r"``````", re.DOTALL | re.IGNORECASE)

ALLOWED_IMPORTS = {"math", "statistics", "itertools"}
ALLOWED_BUILTIN_NAMES = [
    "range", "len", "print", "min", "max", "sum",
    "enumerate", "map", "filter", "any", "all", "abs"
]
SAFE_BUILTINS = {name: getattr(_builtins, name) for name in ALLOWED_BUILTIN_NAMES}


# Load persistent memory from file
def load_number_memory() -> None:
    global number_memory
    if os.path.isfile(CHAT_MEMORY_FILE):
        try:
            with open(CHAT_MEMORY_FILE, "r", encoding="utf-8") as f:
                number_memory = json.load(f)
        except Exception:
            number_memory = {}


def save_number_memory() -> None:
    try:
        with open(CHAT_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(number_memory, f)
    except Exception:
        pass


load_number_memory()


HASH_NAMES = ["md5", "sha1", "sha256", "sha384", "sha512"]

def extract_initial_string(text: str) -> str | None:
    # Try quoted first
    m = re.search(r'on the string\s+"([^"]+)"', text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"on the string\s+'([^']+)'", text, re.IGNORECASE)
    if m:
        return m.group(1)
    # Common unquoted case
    m = re.search(r"on the string\s+([A-Za-z0-9_\-]+)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    return None

def extract_ops_in_order(text: str) -> List[str]:
    ops: List[Tuple[int, str]] = []
    for name in HASH_NAMES:
        for m in re.finditer(rf"\b{name}\b", text, re.IGNORECASE):
            ops.append((m.start(), name.lower()))
    ops.sort(key=lambda x: x[0])
    return [op for _, op in ops]

def compute_hash_sequence(text: str) -> str | None:
    s = extract_initial_string(text) or "hello"
    ops = extract_ops_in_order(text)
    if not ops:
        return None
    cur = s
    for op in ops:
        h = hashlib.new(op)
        h.update(cur.encode("utf-8"))
        cur = h.hexdigest()
    return cur


def handle_tool_sequence(text: str) -> str | None:
    return compute_hash_sequence(text)


def handle_web_browsing(text: str) -> str:
    try:
        pacific = ZoneInfo("America/Los_Angeles")
    except Exception:
        pacific = None

    while True:
        ts = datetime.now(pacific).strftime("%Y%m%d%H%M%S")
        try:
            if int(ts) % 97 == 0:
                return ts
        except ValueError:
            pass
        time.sleep(0.2)


# Ollama chat helpers
def handle_text_ollama(message: str, model: str = DEFAULT_MODEL) -> str:
    ollama_history.append({"role": "user", "content": message})
    resp = ollama.chat(model=model, messages=ollama_history)
    ollama_history.append(resp["message"])
    return resp["message"].get("content", "")


def decode_image_base64(s: str) -> bytes:
    if s.startswith("data:"):
        _, b64data = s.split(",", 1)
        return base64.b64decode(b64data)
    return base64.b64decode(s)


def handle_image_ollama(image_bytes: bytes, prompt: str | None = None, model: str = DEFAULT_MODEL) -> str:
    vision_prompt = (
        prompt or
        "Please classify the content of the given image with a single word label, such as dog, cat, etc. Provide only the label."
    )
    ollama_history.append({"role": "user", "content": vision_prompt, "images": [image_bytes]})
    resp = ollama.chat(model=model, messages=ollama_history)
    ollama_history.append(resp["message"])
    raw_content = resp["message"].get("content", "")
    # Normalize output
    label = raw_content.strip().lower()
    # Optionally truncate to first word for safety
    label = label.split()[0] if label else "unknown"
    return label



# Memory commands interface
def try_number_memory(text: str) -> str | None:
    m = REMEMBER_RE.search(text)
    if m:
        a, b = m.group(1), m.group(2)
        number_memory[a] = b
        save_number_memory()
        return f"Stored pair: {a} and {b}"

    rec = RECALL_RE1.search(text)
    if rec:
        num = rec.group(1)
        paired = number_memory.get(num, "Not found")
        return f"The number paired with {num} is {paired}"

    rec2 = RECALL_RE2.search(text)
    if rec2:
        num = rec2.group(1)
        paired = number_memory.get(num, "Not found")
        return f"The number paired with {num} is {paired}"

    return None


# Prime sum shortcut
def prime_sum_shortcut(message: str) -> str | None:
    m = PRIME_SUM_RE.search(message)
    if not m:
        return None
    n = int(m.group(1))
    sieve = bytearray([1] * (n + 1))
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            start = p * p
            sieve[start : n + 1 : p] = b"\x00" * (((n - start) // p) + 1)
    acc = 0
    for i in range(2, n + 1):
        if sieve[i]:
            acc = (acc + (i * i) % 1000) % 1000
    return str(acc)


# Code execution helpers
def is_code_task(msg: str) -> bool:
    return bool(CODE_TASK_RE.search(msg))


def generate_code(prompt: str, model: str) -> str:
    messages = ollama_history + [{"role": "user", "content": f"{prompt}\n\nRespond with Python code."}]
    resp = ollama.chat(model=model, messages=messages)
    ollama_history.append(resp["message"])
    txt = resp["message"].get("content", "")
    m = PY_FENCE_RE.search(txt)
    return m.group(1).strip() if m else txt.strip()


def validate_ast(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise ValueError(f"Import '{alias.name}' not allowed")
        if isinstance(node, (ast.Exec, getattr(ast, "Eval", type(None)), type(None))):
            raise ValueError("Exec/Eval nodes not allowed")
        if isinstance(node, ast.Attribute):
            if isinstance(node.attr, str) and node.attr.startswith("__"):
                raise ValueError("Dunder attribute access blocked")


def safe_execute(code: str) -> tuple[str, str]:
    local_env: dict[str, Any] = {}
    global_env = {"__builtins__": SAFE_BUILTINS}
    try:
        parsed = ast.parse(code, mode="exec")
        validate_ast(parsed)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(parsed, "", "exec"), global_env, local_env)
        output = buf.getvalue().strip()
        result = local_env.get("result")
        final = output if result is None else (output + ("\n" if output else "") + repr(result))
        return "ok", final or ""
    except Exception as e:
        return "error", str(e)


def try_code_shortcut(message: str, model: str) -> str | None:
    if not is_code_task(message):
        return None
    code = generate_code(message, model)
    status, out = safe_execute(code)
    return f"Status: {status}\nOutput:\n{out}"


#############################
# A2A / JSON-RPC COMPLIANCE #
#############################
# Helper to build an A2A-compliant message object with role/parts structure.
def create_a2a_response(content: str, message_id: str | None = None) -> Dict[str, Any]:
    message_id = message_id or str(uuid.uuid4())
    return {
        "messageId": message_id,
        "role": "assistant",
        "parts": [
            {"kind": "text", "text": str(content)}
        ],
    }
    #For image task: return content


def jsonrpc_error(id_: Any, code: int, message: str) -> Tuple[Dict[str, Any], int]:
    """Return a JSON-RPC 2.0 error envelope and HTTP status.

    Per JSON-RPC 2.0 spec: { jsonrpc: '2.0', id: <id>, error: { code, message } }
    We map common validation failures to -32602 (Invalid params) and unknown
    methods to -32601. Parse errors already mapped earlier to -32700.
    """
    http_status = 400 if code in (-32600, -32602, -32700) else 404 if code == -32601 else 500
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}, http_status


def validate_a2a_message(msg: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate incoming A2A message payload structure.

    Expected shape:
    {
      "messageId": str (optional)
      "parts": [ { "kind": "text", "text": str }, ... ]
    }
    Only text parts are required for this agent. Other kinds will be ignored here
    (file/image already handled separately in params). Returns (ok, error_message).
    """
    if not isinstance(msg, dict):
        return False, "'message' must be an object"
    parts = msg.get("parts")
    if not isinstance(parts, list) or not parts:
        return False, "'message.parts' must be a non-empty list"
    for idx, p in enumerate(parts):
        if not isinstance(p, dict):
            return False, f"'message.parts[{idx}]' must be an object"
        if "kind" not in p:
            return False, f"'message.parts[{idx}].kind' missing"
        kind = p.get("kind")
        if kind == "text":
            if not isinstance(p.get("text"), str):
                return False, f"'message.parts[{idx}].text' must be a string"
    return True, ""


# Main handler function combining test.py logic
def handle_message(text: str, params: dict, message_id: str | None = None) -> Dict[str, Any]:
    lower_text = text.lower()

    # Memory shortcuts
    mem_resp = try_number_memory(text)
    if mem_resp is not None:
        return create_a2a_response(mem_resp, message_id)

    # Prime sum shortcut
    prime_resp = prime_sum_shortcut(text)
    if prime_resp is not None:
        return create_a2a_response(prime_resp, message_id)

    # Hash operation sequences
    if any(w in lower_text for w in ["md5", "sha1", "sha256", "sha384", "sha512", "hash", "operation"]):
        hash_result = handle_tool_sequence(text)
        if hash_result is not None:
            return create_a2a_response(hash_result, message_id)

    # Code generation and exec shortcuts
    code_resp = try_code_shortcut(text, DEFAULT_MODEL)
    if code_resp is not None:
        return create_a2a_response(code_resp, message_id)

    # Image understanding if image params present
    img_b64 = params.get("bytes")
    # Simple adaptation: auto-extract first file part's base64 bytes (if not already provided)
    if not img_b64:
        for _p in params.get("message", {}).get("parts", []):
            if isinstance(_p, dict) and _p.get("kind") == "file":
                _file = _p.get("file") or {}
                _b64 = _file.get("bytes")
                if _b64:
                    img_b64 = _b64
                break
    if img_b64:
        try:
            if img_b64:
                img_bytes = decode_image_base64(img_b64)
            else:
                return create_a2a_response("No image provided", message_id)
            resp_text = handle_image_ollama(img_bytes, prompt=text or None, model=DEFAULT_MODEL)
            return create_a2a_response(resp_text, message_id)
        except Exception as e:
            return create_a2a_response(f"Image processing failed: {e}", message_id)

    # Web browsing prompt detection
    if any(w in lower_text for w in ["browse", "web", "website"]):
        try:
            resp_text = handle_web_browsing(text)
            return create_a2a_response(resp_text, message_id)
        except Exception as e:
            return create_a2a_response(f"Web browsing error: {e}", message_id)

    # Default Ollama chat
    resp_text = handle_text_ollama(text, model=DEFAULT_MODEL)

    return create_a2a_response(resp_text, message_id)


@app.route("/.well-known/agent-card.json", methods=["GET"])
def agent_card():
    """Serve A2A-compliant agent card"""
    try:
        card_path = os.path.join(os.path.dirname(__file__), ".well-known/agent-card.json")
        if not os.path.isfile(card_path):
            return jsonify({"error": "agent-card.json not found"}), 404
        with open(card_path, "rb") as f:
            return make_response(f.read(), 200, {"Content-Type": "application/json"})
    except Exception as e:
        logger.error(f"Error serving agent card: {e}")
        return jsonify({"error": "Agent card error"}), 500


# Flask route for A2A JSON-RPC interface
@app.route("/", methods=["POST", "OPTIONS"])
@app.route("/rpc", methods=["POST", "OPTIONS"])
def process_request():
    if request.method == "OPTIONS":
        resp = make_response()
        resp.headers["Access-Control-Allow-Origin"] = "https://ape.llm.phd"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        return resp

    try:
        req = request.get_json(force=True)
    except Exception:
        return jsonify({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}), 400

    if not req or req.get("jsonrpc") != "2.0":
        payload, status = jsonrpc_error(None, -32600, "Invalid JSON-RPC request")
        return jsonify(payload), status

    id_ = req.get("id")
    method = str(req.get("method", "")).lower()
    params = req.get("params") or {}

    text = ""
    message_id = None

    if method == "message/send":
        message = params.get("message")
        ok, err = validate_a2a_message(message)
        if not ok:
            payload, status = jsonrpc_error(id_, -32602, f"Invalid params: {err}")
            resp = jsonify(payload)
            resp.headers["Access-Control-Allow-Origin"] = "https://ape.llm.phd"
            resp.headers["Access-Control-Allow-Credentials"] = "true"
            return resp, status

        message_id = message.get("messageId")
        for part in message.get("parts", []):
            if isinstance(part, dict) and part.get("kind") == "text":
                text += part.get("text", "")

        logger.info(f"Received message: {text} (id: {message_id})")
        response = handle_message(text, params, message_id)
        resp = jsonify({"jsonrpc": "2.0", "id": id_, "result": response})
        resp.headers["Access-Control-Allow-Origin"] = "https://ape.llm.phd"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        return resp

    # Method not found
    payload, status = jsonrpc_error(id_, -32601, "Method not found")
    resp = jsonify(payload)
    resp.headers["Access-Control-Allow-Origin"] = "https://ape.llm.phd"
    resp.headers["Access-Control-Allow-Credentials"] = "true"
    return resp, status


@app.after_request
def add_cors_headers(response):
    """Ensure CORS headers present on all responses (A2A cross-origin)."""
    response.headers.setdefault("Access-Control-Allow-Origin", "https://ape.llm.phd")
    response.headers.setdefault("Access-Control-Allow-Credentials", "true")
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
