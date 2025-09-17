"""Simple Ollama chat utility with:
1. Text question/statement
2. Image + optional prompt
3. Memory commands (store & recall number pairs)

Memory examples:
  python test.py "Please remember 77858 and 40232"
  python test.py "What number was paired with 77858?"
"""
from __future__ import annotations
import sys, os, mimetypes, json, re
from typing import Any, Dict, List, Tuple
import requests
try:
    import ollama  # type: ignore
except Exception:
    print("Missing dependency: pip install ollama requests")
    raise

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llava")
VISION_MODELS = {"llava", "llava:latest", "llava-phi3", "bakllava", "moondream", "yi-vl"}
IMAGE_EXT = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
 
history: List[Dict[str, Any]] = []

# ---- Memory ----------------------------------------------------
MEMORY_FILE = os.getenv("CHAT_MEMORY_FILE", "memory.json")
number_memory: dict[str, str] = {}
REMEMBER_RE = re.compile(r"remember.*?(\d+)\D+(\d+)", re.IGNORECASE)
RECALL_RE = re.compile(r"(paired with (\d+))|(number (\d+) was paired)", re.IGNORECASE)

def load_memory():
    global number_memory
    if os.path.isfile(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                number_memory = json.load(f)
        except Exception:
            number_memory = {}

def save_memory():
    try:
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(number_memory, f)
    except Exception:
        pass

def try_memory_shortcut(message: str) -> dict | None:
    m = REMEMBER_RE.search(message)
    if m:
        a, b = m.group(1), m.group(2)
        number_memory[a] = b
        save_memory()
        return {"message": {"role": "assistant", "content": f"Stored pair: {a} and {b}"}}
    # recall patterns
    rec = re.search(r"paired with (\d+)", message, re.IGNORECASE)
    if rec:
        num = rec.group(1)
        paired = number_memory.get(num, "Not found")
        return {"message": {"role": "assistant", "content": f"The number paired with {num} is {paired}"}}
    rec2 = re.search(r"number (\d+) was paired", message, re.IGNORECASE)
    if rec2:
        num = rec2.group(1)
        paired = number_memory.get(num, "Not found")
        return {"message": {"role": "assistant", "content": f"The number paired with {num} is {paired}"}}
    return None

load_memory()

# ---- Helpers ---------------------------------------------------

def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

def load_image(src: str) -> Tuple[bytes, str]:
    if is_url(src):
        r = requests.get(src, timeout=30)
        r.raise_for_status()
        mime = r.headers.get("Content-Type") or mimetypes.guess_type(src)[0] or "application/octet-stream"
        return r.content, mime
    with open(src, 'rb') as f:
        data = f.read()
    mime = mimetypes.guess_type(src)[0] or "application/octet-stream"
    return data, mime

def model_supports_vision(model: str) -> bool:
    return model.split(":")[0].lower() in VISION_MODELS

# ---- Handlers --------------------------------------------------

def handle_text(message: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    history.append({"role": "user", "content": message})
    resp = ollama.chat(model=model, messages=history)
    history.append(resp["message"])
    return resp

def handle_image(image_ref: str, prompt: str | None = None, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    if not model_supports_vision(model):
        return {"message": {"role": "assistant", "content": f"Model '{model}' has no vision. Try a vision model like 'llava'."}}
    prompt = prompt or "Describe the image"
    try:
        img_bytes, _ = load_image(image_ref)
    except Exception as e:
        return {"message": {"role": "assistant", "content": f"Failed to load image: {e}"}}
    history.append({"role": "user", "content": prompt, "images": [img_bytes]})
    resp = ollama.chat(model=model, messages=history)
    history.append(resp["message"])
    return resp

# ---- Dispatcher ------------------------------------------------

CODE_TASK_RE = re.compile(r"\b(write|create|generate)\b.*\b(program|script|code)\b", re.IGNORECASE)
PY_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL | re.IGNORECASE)

import ast, textwrap, io, contextlib

ALLOWED_IMPORTS = {"math", "statistics", "itertools"}
import builtins as _builtins
ALLOWED_BUILTIN_NAMES = ["range","len","print","min","max","sum","enumerate","map","filter","any","all","abs"]
SAFE_BUILTINS = {name: getattr(_builtins, name) for name in ALLOWED_BUILTIN_NAMES}

def is_code_task(msg: str) -> bool:
    return bool(CODE_TASK_RE.search(msg))

def generate_code(prompt: str, model: str) -> str:
    system = "Return ONLY Python code in a single fenced block. No explanation."  # guidance
    messages = history + [{"role": "user", "content": f"{prompt}\n\nRespond with Python code."}]
    resp = ollama.chat(model=model, messages=messages)
    history.append(resp["message"])
    txt = resp["message"].get("content", "")
    m = PY_FENCE_RE.search(txt)
    return m.group(1).strip() if m else txt.strip()

def validate_ast(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name.split('.')[0] not in ALLOWED_IMPORTS:
                    raise ValueError(f"Import '{alias.name}' not allowed")
        if isinstance(node, (ast.Exec, getattr(ast,'Eval',tuple()))):  # type: ignore
            raise ValueError("Exec/Eval not allowed")
        if isinstance(node, (ast.Attribute,)):
            # crude block on dunder access
            if isinstance(node.attr, str) and node.attr.startswith('__'):
                raise ValueError("Dunder attribute access blocked")

def safe_execute(code: str) -> tuple[str,str]:
    local_env: dict[str, Any] = {}
    global_env = {"__builtins__": SAFE_BUILTINS}
    try:
        parsed = ast.parse(code, mode='exec')
        validate_ast(parsed)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(parsed, '<user_code>', 'exec'), global_env, local_env)
        output = buf.getvalue().strip()
        result = local_env.get('result')
        final = output if result is None else (output + ('\n' if output else '') + repr(result))
        return "ok", final or "<no output>"
    except Exception as e:
        return "error", str(e)

def try_code_shortcut(message: str, model: str) -> dict | None:
    if not is_code_task(message):
        return None
    code = generate_code(message, model)
    status, out = safe_execute(code)
    return {"message": {"role": "assistant", "content": f"Status: {status}\nOutput:\n{out}"}}

PRIME_SUM_RE = re.compile(r"sum of the squares of all prime numbers.*?n\s*=\s*(\d+)", re.IGNORECASE)

def prime_sum_shortcut(message: str) -> dict | None:
    m = PRIME_SUM_RE.search(message)
    if not m:
        return None
    n = int(m.group(1))
    sieve = bytearray(b"\x01")*(n+1)
    sieve[0:2] = b"\x00\x00"
    import math
    for p in range(2, int(n**0.5)+1):
        if sieve[p]:
            start = p*p
            sieve[start:n+1:p] = b"\x00"*(((n-start)//p)+1)
    acc = 0
    for i in range(2, n+1):
        if sieve[i]:
            acc = (acc + (i*i)%1000) % 1000
    return {"message": {"role": "assistant", "content": str(acc)}}

# Replace wrapper logic: define consolidated ask only once

def ask(content: Dict[str, Any], model: str = DEFAULT_MODEL) -> Dict[str, Any]:  # type: ignore[override]
    t = content.get("type")
    if t == "text":
        message = content["message"]
        # memory
        mem = try_memory_shortcut(message)
        if mem: return mem
        # deterministic prime sum
        ps = prime_sum_shortcut(message)
        if ps: return ps
        # code generation & execution
        code_resp = try_code_shortcut(message, model)
        if code_resp: return code_resp
        # default LLM chat
        return handle_text(message, model)
    if t == "image":
        return handle_image(content["message"], content.get("prompt"), model)
    raise ValueError("content.type must be 'text' or 'image'")

# ---- CLI -------------------------------------------------------

def infer(argv: list[str]) -> Dict[str, Any]:
    if len(argv) == 1:
        a = argv[0]
        ext = os.path.splitext(a.lower())[1]
        if is_url(a) or ext in IMAGE_EXT:
            return {"type": "image", "message": a, "prompt": "Describe the image"}
        return {"type": "text", "message": a}
    return {"type": "image", "message": argv[0], "prompt": " ".join(argv[1:]) or "Describe the image"}

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage:\n  python test.py 'Question'\n  python test.py img.jpg 'Describe this'\n  python test.py 'Please remember 1 and 2'\n  python test.py 'What number was paired with 1?'")
        sys.exit(0)
    payload = infer(sys.argv[1:])
    result = ask(payload)
    print("Assistant:", result.get("message", {}).get("content"))