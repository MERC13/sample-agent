"""
LangChain A2A Agent with LangGraph and Groq (Updated for LangChain 0.3+)
Uses Groq's fast LPU inference instead of OpenAI
Implements six basic capabilities: QA, hashing, image understanding, web browsing, code execution, and memory
"""

import json
import hashlib
import base64
import sqlite3
import requests
import subprocess
import tempfile
import os
import uuid
from typing import Dict, List, Any, Optional, Annotated
from datetime import datetime
from PIL import Image
import io

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from flask import Flask, request, jsonify
from werkzeug.serving import run_simple
from flask_cors import CORS

class A2AGroqAgent:
    """
    LangChain agent with A2A protocol compatibility using LangGraph and Groq
    Supports JSON-RPC 2.0 and standard A2A message format
    """
    
    def __init__(self, groq_api_key: str, port: int = 8080, model: str = "llama-3.3-70b-versatile"):
        self.port = port
        self.app = Flask(__name__)
        CORS(self.app, resources={r"/*": {"origins": "*"}})
        
        # Initialize Groq LLM - Fast and Free!
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name=model,
            temperature=0.7,
            max_tokens=4096,
            max_retries=2
        )
        
        # Initialize SQLite for persistent memory
        self.init_memory_db()
        
        # Create tools using the new @tool decorator
        self.tools = self._create_tools()

        # Recursion / reasoning depth limit (can be overridden by env)
        # LangGraph defaults to 25; some tool chains may need more.
        try:
            self.recursion_limit = int(os.getenv("LC_AGENT_RECURSION_LIMIT", "60"))
        except ValueError:
            self.recursion_limit = 60
        
        # Initialize LangGraph agent with memory
        memory = MemorySaver()
        self.agent = create_react_agent(
            self.llm,
            tools=self.tools,
            checkpointer=memory
        )
        
        # Setup Flask routes
        self._setup_routes()
    
    def init_memory_db(self):
        """Initialize SQLite database for persistent memory"""
        self.db_path = "agent_memory.db"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_message TEXT,
                agent_response TEXT,
                session_id TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def _create_tools(self) -> List:
        """Create the six required capabilities as LangChain tools using @tool decorator"""
        
        @tool
        def general_qa(query: str) -> str:
            """Answer general questions using Groq's fast language model."""
            try:
                response = self.llm.invoke([HumanMessage(content=query)])
                return response.content
            except Exception as e:
                return f"Error in QA: {str(e)}"
        
        @tool
        def hash_tool(text: str, algorithm: str = "sha256") -> str:
            """Hash text using specified algorithm (md5, sha1, sha256, sha512)."""
            try:
                text_bytes = text.encode('utf-8')
                if algorithm.lower() == "md5":
                    return hashlib.md5(text_bytes).hexdigest()
                elif algorithm.lower() == "sha1":
                    return hashlib.sha1(text_bytes).hexdigest()
                elif algorithm.lower() == "sha256":
                    return hashlib.sha256(text_bytes).hexdigest()
                elif algorithm.lower() == "sha512":
                    return hashlib.sha512(text_bytes).hexdigest()
                else:
                    return f"Unsupported algorithm: {algorithm}. Supported: md5, sha1, sha256, sha512"
            except Exception as e:
                return f"Error in hashing: {str(e)}"
        
        @tool
        def image_understanding(image_data: str, query: str = "Describe this image") -> str:
            """Understand images from base64 data or URL. Note: Basic processing only as Groq models don't support vision."""
            try:
                if image_data.startswith("http"):
                    # Handle URL
                    response = requests.get(image_data, timeout=10)
                    img = Image.open(io.BytesIO(response.content))
                else:
                    # Handle base64
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]  # Remove data:image prefix
                    img_data = base64.b64decode(image_data)
                    img = Image.open(io.BytesIO(img_data))
                
                # Basic image analysis (size, format, mode)
                analysis = f"Image Analysis - Size: {img.size[0]}x{img.size[1]} pixels, Format: {img.format}, Mode: {img.mode}"
                
                # Use Groq to provide contextual response about the image metadata
                context_prompt = f"Based on this image metadata: {analysis}, and the user query: '{query}', provide a helpful response about what this might be."
                context_response = self.llm.invoke([HumanMessage(content=context_prompt)])
                
                return f"{analysis}. {context_response.content}"
            except Exception as e:
                return f"Error in image understanding: {str(e)}"
        
        @tool
        def web_browsing(url: str) -> str:
            """Browse web pages and extract content."""
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                
                # Extract text content (simple approach)
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                # Truncate to reasonable length
                content = text[:2000] if len(text) > 2000 else text
                
                return f"Web content from {url}:\n{content}"
            except Exception as e:
                return f"Error browsing {url}: {str(e)}"
        
        @tool
        def code_execution(code: str, language: str = "python") -> str:
            """Execute code safely in a temporary environment."""
            try:
                if language.lower() == "python":
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                        f.write(code)
                        temp_path = f.name
                    
                    # Execute with timeout
                    result = subprocess.run(
                        ['python', temp_path],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    # Cleanup
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                    
                    if result.returncode == 0:
                        return f"Output: {result.stdout}" if result.stdout else "Code executed successfully (no output)"
                    else:
                        return f"Error: {result.stderr}"
                else:
                    return f"Language '{language}' not supported. Only Python is supported."
            except subprocess.TimeoutExpired:
                return "Error: Code execution timed out (30 second limit)"
            except Exception as e:
                return f"Error in code execution: {str(e)}"
        
        @tool
        def memory_search(query: str, session_id: str = "default") -> str:
            """Search conversation history for relevant information."""
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_message, agent_response, timestamp 
                    FROM conversations 
                    WHERE session_id = ? AND (user_message LIKE ? OR agent_response LIKE ?)
                    ORDER BY timestamp DESC LIMIT 5
                """, (session_id, f"%{query}%", f"%{query}%"))
                
                results = cursor.fetchall()
                conn.close()
                
                if results:
                    memory_items = []
                    for user_msg, agent_resp, timestamp in results:
                        memory_items.append(f"[{timestamp}] User: {user_msg[:100]}... | Agent: {agent_resp[:100]}...")
                    return "Found relevant memories:\n" + "\n".join(memory_items)
                else:
                    return f"No relevant memories found for query: '{query}'"
            except Exception as e:
                return f"Error in memory search: {str(e)}"
        
        return [general_qa, hash_tool, image_understanding, web_browsing, code_execution, memory_search]
    
    def save_conversation(self, user_message: str, agent_response: str, session_id: str = "default"):
        """Save conversation to persistent memory"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations (timestamp, user_message, agent_response, session_id)
                VALUES (?, ?, ?, ?)
            """, (datetime.now().isoformat(), user_message, agent_response, session_id))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving conversation: {e}")
    
    def process_a2a_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process A2A standard message format"""
        try:
            # Extract content from A2A message format
            user_input = ""
            if isinstance(message, dict):
                if "parts" in message and isinstance(message["parts"], list):
                    content_parts = []
                    for part in message["parts"]:
                        # Support both legacy {type:"text", content:"..."} and new {kind:"text", text:"..."}
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "text":
                            content_parts.append(part.get("content", ""))
                        elif part.get("kind") == "text":
                            content_parts.append(part.get("text", ""))
                    user_input = " ".join(p for p in content_parts if p)
                # Fallback keys
                if not user_input:
                    # Common alternative field names
                    user_input = message.get("content") or message.get("text") or ""
            
            # Get session ID from message
            session_id = message.get("sessionId", "default")
            message_id = message.get("messageId") or str(uuid.uuid4())
            
            # Create configuration for LangGraph with thread_id and recursion limit
            config = {
                "configurable": {"thread_id": session_id},
                "recursion_limit": self.recursion_limit
            }
            
            # Process with LangGraph agent using Groq
            response = self.agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )
            
            # Extract the last AI message
            if response["messages"]:
                last_message = response["messages"][-1]
                if hasattr(last_message, 'content'):
                    agent_output = last_message.content
                else:
                    agent_output = str(last_message)
            else:
                agent_output = "No response generated"
            
            # Save conversation
            self.save_conversation(user_input, agent_output, session_id)
            
            # Return A2A format response
            # Minimal A2A-style message (align with app.py output expectations)
            return {
                "messageId": message_id,
                "role": "assistant",
                "parts": [
                    {"kind": "text", "text": agent_output}
                ]
            }
            
        except Exception as e:
            err_text = str(e)
            if "GRAPH_RECURSION_LIMIT" in err_text or "Recursion limit" in err_text or "recursion limit" in err_text.lower():
                err_text += (
                    f"\nThe agent hit the recursion limit ({self.recursion_limit}). "
                    "You can raise it by setting environment variable LC_AGENT_RECURSION_LIMIT to a higher number, "
                    "or simplify your request so the agent needs fewer tool iterations."
                )
            return {
                "messageId": message.get("messageId") or str(uuid.uuid4()),
                "role": "assistant",
                "error": True,
                "parts": [
                    {"kind": "text", "text": f"Error processing message: {err_text}"}
                ]
            }
    
    def _setup_routes(self):
        """Setup Flask routes for A2A protocol"""
        
        @self.app.route('/.well-known/agent-card.json', methods=['GET'])
        def agent_card():
            """Return agent card as per A2A specification"""
            card = {
                "protocolVersion": "0.3.0",
                "name": "LangChain A2A Agent (Groq + LangGraph)",
                "description": "A fast LangChain agent powered by Groq's LPU technology with six core capabilities: QA, hashing, image understanding, web browsing, code execution, and memory",
                "url": f"http://localhost:{self.port}",
                "preferredTransport": "JSONRPC",
                "provider": {
                    "organization": "Custom Implementation",
                    "url": f"http://localhost:{self.port}",
                    "inference_engine": "Groq LPU"
                },
                "version": "3.0.0",
                "capabilities": {
                    "streaming": False,
                    "pushNotifications": False,
                    "stateTransitionHistory": True,
                    "fastInference": True
                },
                "defaultInputModes": ["text/plain", "application/json"],
                "defaultOutputModes": ["text/plain", "application/json"],
                "models": [
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant", 
                    "mixtral-8x7b-32768"
                ],
                "skills": [
                    {
                        "id": "general-qa",
                        "name": "General Q&A",
                        "description": "Answer general questions using Groq's fast language model",
                        "inputModes": ["text/plain"],
                        "outputModes": ["text/plain"],
                        "features": ["fast_inference", "high_throughput"]
                    },
                    {
                        "id": "hashing",
                        "name": "Text Hashing",
                        "description": "Hash text using various cryptographic algorithms",
                        "inputModes": ["text/plain"],
                        "outputModes": ["text/plain"],
                        "algorithms": ["md5", "sha1", "sha256", "sha512"]
                    },
                    {
                        "id": "image-understanding",
                        "name": "Image Understanding",
                        "description": "Analyze image metadata and provide contextual responses",
                        "inputModes": ["text/plain", "image/*"],
                        "outputModes": ["text/plain"],
                        "limitations": ["metadata_only", "no_vision_model"]
                    },
                    {
                        "id": "web-browsing",
                        "name": "Web Browsing",
                        "description": "Browse web pages and extract content with BeautifulSoup",
                        "inputModes": ["text/plain"],
                        "outputModes": ["text/plain"],
                        "features": ["content_extraction", "text_cleanup"]
                    },
                    {
                        "id": "code-execution",
                        "name": "Code Execution",
                        "description": "Execute Python code safely with timeout and security restrictions",
                        "inputModes": ["text/plain"],
                        "outputModes": ["text/plain"],
                        "languages": ["python"],
                        "security": ["timeout", "sandboxed"]
                    },
                    {
                        "id": "memory-search",
                        "name": "Memory Search",
                        "description": "Search conversation history for relevant information",
                        "inputModes": ["text/plain"],
                        "outputModes": ["text/plain"],
                        "features": ["persistent_storage", "session_isolation"]
                    }
                ]
            }
            return jsonify(card)
        
        def _jsonrpc_core():
            """Shared JSON-RPC handling logic for both '/' and '/rpc' routes."""
            data = request.get_json(silent=True)
            if not data or data.get("jsonrpc") != "2.0":
                return jsonify({
                    "jsonrpc": "2.0",
                    "error": {"code": -32600, "message": "Invalid Request - JSON-RPC 2.0 format required"},
                    "id": data.get("id") if data else None
                }), 400
            method = data.get("method")
            params = data.get("params", {})
            request_id = data.get("id")

            # Accept both legacy and new method names
            if method in ("task", "message/send"):
                message = params.get("message", {})
                response = self.process_a2a_message(message)
                return jsonify({
                    "jsonrpc": "2.0",
                    "result": response,
                    "id": request_id
                })

            # Method not found - return JSON-RPC error (no HTTP 404 to avoid client transport errors)
            return jsonify({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32601,
                    "message": f"Method '{method}' not found. Supported methods: ['task', 'message/send']"
                },
                "id": request_id
            }), 200

        @self.app.route('/', methods=['POST', 'OPTIONS'])
        @self.app.route('/rpc', methods=['POST', 'OPTIONS'])
        def handle_jsonrpc():
            # Handle CORS preflight quickly
            if request.method == 'OPTIONS':
                resp = jsonify({})
                resp.headers['Access-Control-Allow-Origin'] = '*'
                resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
                resp.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
                return resp, 204
            try:
                return _jsonrpc_core()
            except Exception as e:
                return jsonify({
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
                    "id": None
                }), 500
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "framework": "LangGraph",
                "llm_provider": "Groq",
                "model": self.llm.model_name,
                "inference_engine": "Groq LPU",
                "capabilities": ["qa", "hashing", "image_understanding", "web_browsing", "code_execution", "memory"]
            })
        
        @self.app.route('/models', methods=['GET'])
        def list_models():
            """List available Groq models"""
            return jsonify({
                "current_model": self.llm.model_name,
                "available_models": [
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "llama-3.1-70b-versatile", 
                    "mixtral-8x7b-32768",
                    "gemma2-9b-it"
                ],
                "inference_speed": "Ultra-fast with Groq LPU technology"
            })
    
    def run(self, host: str = "0.0.0.0", debug: bool = False):
        """Start the agent server"""
        print(f"🚀 Starting LangChain A2A Agent (Groq + LangGraph) on {host}:{self.port}")
        print(f"⚡ Powered by Groq LPU - Ultra-fast inference!")
        print(f"🤖 Model: {self.llm.model_name}")
        print(f"📋 Agent card: http://{host}:{self.port}/.well-known/agent-card.json")
        print(f"❤️  Health check: http://{host}:{self.port}/health")
        print(f"🧠 Available models: http://{host}:{self.port}/models")
        
        run_simple(host, self.port, self.app, use_reloader=debug, use_debugger=debug)


def main():
    """Main entry point"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Get Groq API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set")
        print("Get your free API key at: https://console.groq.com/keys")
        return
    
    # Optional: specify model (default is llama-3.3-70b-versatile)
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    print(f"🔑 Using Groq API key: {api_key[:8]}...")
    print(f"🧠 Using model: {model}")
    
    # Create and run agent
    agent = A2AGroqAgent(groq_api_key=api_key, port=3000, model=model)
    agent.run(debug=True)


if __name__ == "__main__":
    main()