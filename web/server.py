#!/usr/bin/env python3
"""
PS5 Local LLM Web Server, SSE Streaming Proxy & Tool Execution Engine
Serves the Vercel AI Elements chat webpage, proxies SSE stream requests
to the PS5 LLM endpoint (http://ps5.tailnet.example:8081/v1/chat/completions),
and executes local tools (calculator, python, time, fetch).
"""

import sys
import os
import re
import json
import time
import math
import socket
import ssl
import argparse
import datetime
import subprocess
import threading
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

# The console's address on the tailnet and the key it was built with come from the environment.
PS5_HOST = os.environ.get("PS5_HOST", "ps5.tailnet.example")
PS5_BASE = f"http://{PS5_HOST}:8081"
DEFAULT_TARGET_URL = (PS5_BASE + "/v1/chat/completions")
DEFAULT_HEALTH_URL = (PS5_BASE + "/health")
DEFAULT_API_KEY = os.environ.get("PSLLAMA_API_KEY", "")
PORT = 8080

INFERENCE_LOCK = threading.Lock()
LAST_HEALTH = {}

STATIC_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_MODELS = [
    {
        "id": "flux2-klein-4b",
        "name": "FLUX.2-klein-4B Q8_0",
        "description": "FLUX.2-klein-4B Q8_0 + Qwen3-4B Q4_K_M + FLUX.2 VAE",
        "default_width": 768,
        "default_height": 768,
        "default_steps": 4,
        "default_cfg": 1.0,
        "warm_time_s": 17.6,
        "recommended": True,
        "badge": "Recommended"
    },
    {
        "id": "z-image-turbo",
        "name": "Z-Image-Turbo Q8_0",
        "description": "Z-Image-Turbo Q8_0 + Qwen3-4B-Instruct Q4_K_M + Flux VAE",
        "default_width": 768,
        "default_height": 768,
        "default_steps": 8,
        "default_cfg": 1.0,
        "warm_time_s": 42.0,
        "recommended": False,
        "badge": "Best Photorealism"
    },
    {
        "id": "sd-turbo-fp16",
        "name": "SD-Turbo FP16",
        "description": "SD-Turbo FP16 (2.6 GB)",
        "default_width": 512,
        "default_height": 512,
        "default_steps": 1,
        "default_cfg": 1.0,
        "warm_time_s": 0.93,
        "recommended": False,
        "badge": "Sub-Second Draft"
    },
    {
        "id": "z-image-turbo-abliterated",
        "name": "Z-Image-Turbo (Abliterated Encoder)",
        "description": "Z-Image Q8_0 + Abliterated Qwen3-4B encoder",
        "default_width": 768,
        "default_height": 768,
        "default_steps": 8,
        "default_cfg": 1.0,
        "warm_time_s": 42.0,
        "recommended": False,
        "badge": "Uncensored"
    },
    {
        "id": "ernie-image-turbo",
        "name": "ERNIE-Image-Turbo Q6_K",
        "description": "ERNIE-Image-Turbo Q6_K + Ministral-3-3B Q4_K_M",
        "default_width": 768,
        "default_height": 768,
        "default_steps": 8,
        "default_cfg": 1.0,
        "warm_time_s": 66.1,
        "recommended": False,
        "badge": "Stylized Art"
    }
]

VIDEO_MODELS = [
    {
        "id": "wan2.1-t2v-1.3b",
        "name": "Wan2.1 T2V 1.3B",
        "description": "Wan2.1 T2V 1.3B FP16 + umt5-xxl Q4_K_M + Wan2.1 VAE (6.43 GB unified Vulkan)",
        "default_width": 416,
        "default_height": 240,
        "default_frames": 9,
        "default_fps": 16,
        "default_steps": 20,
        "default_cfg": 6.0,
        "warm_time_s": 59.6,
        "recommended": True,
        "badge": "Active / Verified",
        "modes": ["t2v", "i2v"]
    },
    {
        "id": "hunyuanvideo-1.5-i2v-480p",
        "name": "HunyuanVideo 1.5 I2V 480p",
        "description": "HunyuanVideo 1.5 Q4_K_M + Qwen2.5-VL-7B + SigCLIP + VAE (13.6 GB)",
        "default_width": 832,
        "default_height": 480,
        "default_frames": 17,
        "default_fps": 16,
        "default_steps": 20,
        "default_cfg": 6.0,
        "warm_time_s": 120.0,
        "recommended": False,
        "badge": "I2V 480p",
        "modes": ["i2v", "t2v"]
    },
    {
        "id": "wan2.2-ti2v-5b",
        "name": "Wan2.2 TI2V 5B",
        "description": "Wan2.2 TI2V 5B Q8_0 + umt5-xxl encoder + VAE (6.8 GB)",
        "default_width": 416,
        "default_height": 240,
        "default_frames": 9,
        "default_fps": 16,
        "default_steps": 20,
        "default_cfg": 6.0,
        "warm_time_s": 75.0,
        "recommended": False,
        "badge": "Next-Gen 5B",
        "modes": ["t2v", "i2v"]
    }
]

STT_MODELS = [
    {
        "id": "whisper/ggml-large-v3-turbo-q5_0.bin",
        "name": "Whisper Large v3 Turbo (Q5_0)",
        "description": "whisper.cpp on Vulkan0 (10× real-time, word-perfect transcription)",
        "load_time_s": 6.4,
        "speed_factor": "10× Real-Time",
        "recommended": True,
        "badge": "Recommended · 10× RT"
    },
    {
        "id": "whisper/ggml-tiny.en.bin",
        "name": "Whisper Tiny.en",
        "description": "whisper.cpp tiny English draft (0.21s latency)",
        "load_time_s": 0.96,
        "speed_factor": "38× Real-Time",
        "recommended": False,
        "badge": "Ultra-Fast Draft"
    }
]

TTS_MODELS = [
    {
        "id": "qwen3-tts-1.7b",
        "name": "Qwen3-TTS 1.7B Base",
        "description": "llama.cpp mtmd audio generation on Vulkan0 (24 kHz, ~2× real-time, zero-shot voice cloning)",
        "load_time_s": 15.5,
        "sample_rate": 24000,
        "speed_factor": "~2× Real-Time",
        "supports_cloning": True,
        "recommended": True,
        "badge": "Voice Clone Ready"
    }
]

# ============================================================================
# Tool Implementations
# ============================================================================

def tool_calculator(expression: str) -> dict:
    """Safely evaluates a mathematical expression."""
    clean_expr = expression.strip()
    safe_dict = {
        "__builtins__": None,
        "math": math,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
        "floor": math.floor,
        "ceil": math.ceil,
    }
    # Basic sanitize check
    if any(k in clean_expr for k in ["import", "exec", "eval", "os", "sys", "open", "__"]):
        raise ValueError("Invalid mathematical expression")
    res = eval(clean_expr, safe_dict, {})
    return {"expression": clean_expr, "result": res}

def tool_get_current_time() -> dict:
    """Returns current real-world date and time."""
    now_local = datetime.datetime.now()
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    return {
        "local_time": now_local.strftime("%Y-%m-%d %H:%M:%S"),
        "utc_time": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "day_of_week": now_local.strftime("%A"),
        "timezone": time.tzname[0] if time.tzname else "Local"
    }

def tool_python_interpreter(code: str) -> dict:
    """Executes a Python 3 code snippet in a subprocess with a timeout."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=5
        )
        return {
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "exit_code": proc.returncode
        }
    except subprocess.TimeoutExpired:
        return {"error": "Execution timed out after 5.0 seconds"}

def tool_fetch_web(url: str) -> dict:
    """Fetches text content from a web URL."""
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "https://" + url
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "AIElements-Agent/1.0 (PS5-Local-LLM)"}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        content_type = resp.headers.get("Content-Type", "")
        raw_bytes = resp.read(64 * 1024) # Cap at 64KB
        text = raw_bytes.decode("utf-8", errors="replace")
        
        # If HTML, strip tags for a readable summary
        if "html" in content_type.lower():
            text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
        
        return {
            "url": url,
            "status": resp.status,
            "content_preview": text[:1500]
        }

AVAILABLE_TOOLS = {
    "calculator": tool_calculator,
    "get_current_time": tool_get_current_time,
    "python_interpreter": tool_python_interpreter,
    "fetch_web": tool_fetch_web,
}

# ============================================================================
# HTTP Request Handler
# ============================================================================

class ChatProxyHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError, ssl.SSLError) as e:
            sys.stderr.write(f"[HTTP/SSL Disconnect] Socket error in handle_one_request: {e}\n")
            self.close_connection = True

    def log_message(self, format, *args):
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

    def end_headers(self):
        if self.command == "GET" and not self.path.startswith(("/api/", "/v1/")):
            self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept, Authorization, X-Requested-With, X-Target-Endpoint")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        if self.path in ["/v1/models", "/models", "/api/models"]:
            self.handle_models()
        elif self.path in ["/api/images/models", "/v1/images/models", "/models/images"]:
            self.handle_list_image_models()
        elif self.path in ["/api/videos/models", "/v1/videos/models", "/models/videos"]:
            self.handle_list_video_models()
        elif self.path in ["/api/stt/models", "/v1/audio/transcriptions/models"]:
            self.handle_list_stt_models()
        elif self.path in ["/api/tts/models", "/v1/audio/speech/models"]:
            self.handle_list_tts_models()
        elif self.path in ["/api/audio/models", "/v1/audio/models"]:
            self.handle_list_all_audio_models()
        elif self.path in ["/api/health", "/health", "/v1/health"]:
            self.handle_health()
        elif self.path == "/api/tools":
            self.handle_list_tools()
        else:
            if self.path == "/":
                self.path = "/index.html"
            super().do_GET()

    def do_POST(self):
        if self.path in ["/api/chat", "/v1/chat/completions", "/chat/completions"]:
            self.handle_chat_proxy()
        elif self.path in ["/api/models/load", "/v1/models/load", "/models/load"]:
            self.handle_model_load()
        elif self.path in ["/api/images/generate", "/api/images/generations", "/v1/images/generations"]:
            self.handle_image_generate()
        elif self.path in ["/api/videos/generate", "/api/videos/generations", "/v1/videos/generations"]:
            self.handle_video_generate()
        elif self.path in ["/api/audio/transcriptions", "/v1/audio/transcriptions"]:
            self.handle_audio_transcribe()
        elif self.path in ["/api/audio/speech", "/v1/audio/speech"]:
            self.handle_audio_speech()
        elif self.path in ["/api/voice/chat", "/v1/voice/chat"]:
            self.handle_voice_chat_proxy()
        elif self.path == "/api/execute_tool":
            self.handle_execute_tool()
        else:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Not Found"}')

    def handle_models(self):
        """Proxy models list from PS5 with formatted sizes and loaded status."""
        try:
            req = urllib.request.Request(
                (PS5_BASE + "/v1/models"),
                headers={"Authorization": f"Bearer {DEFAULT_API_KEY}"}
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                raw_data = json.loads(resp.read().decode("utf-8"))
                models = raw_data.get("data", [])
                for m in models:
                    size = m.get("size_bytes", 0)
                    if size >= 1024 * 1024 * 1024:
                        m["size_formatted"] = f"{size / (1024**3):.2f} GB"
                    elif size >= 1024 * 1024:
                        m["size_formatted"] = f"{size / (1024**2):.1f} MB"
                    else:
                        m["size_formatted"] = f"{size} B"
                
                resp_bytes = json.dumps({"object": "list", "data": models}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)
                return
        except Exception as e:
            sys.stderr.write(f"[Proxy Error] handle_models failed: {e}\n")

        # Fallback list
        data = json.dumps({
            "object": "list",
            "data": [
                {"id": "ps5-local", "name": "ps5-local", "loaded": True, "size_formatted": "6.26 GB"},
                {"id": "gemma-3-12b-it-Q4_K_M.gguf", "name": "gemma-3-12b", "loaded": False, "size_formatted": "6.80 GB"},
                {"id": "Qwen3.8-27B-UD-IQ2_S.gguf", "name": "qwen38-27b", "loaded": False, "size_formatted": "7.80 GB"}
            ]
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def handle_model_load(self):
        """Proxy model load request to PS5 backend with thread safety and extended timeout."""
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            payload = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Invalid JSON body"}')
            return

        # Normalize parameters if provided in nested dict or aliases
        if "parameters" in payload and isinstance(payload["parameters"], dict):
            for k, v in payload["parameters"].items():
                payload.setdefault(k, v)
        if "n_ctx" in payload:
            payload.setdefault("context_tokens", payload["n_ctx"])
        if "max_tokens" in payload:
            payload.setdefault("max_output_tokens", payload["max_tokens"])
        if "n_gpu_layers" in payload:
            payload.setdefault("gpu_layers", payload["n_gpu_layers"])

        target_url = (PS5_BASE + "/v1/models/load")
        is_stream = bool(payload.get("stream", False))
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEFAULT_API_KEY}"
        }
        if is_stream:
            headers["Accept"] = "text/event-stream"

        # Use inference lock so we don't swap models mid-generation
        with INFERENCE_LOCK:
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(
                    target_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=600) as resp:
                    resp_content_type = resp.headers.get("Content-Type", "")
                    if is_stream or "text/event-stream" in resp_content_type:
                        self.send_response(resp.status)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("X-Accel-Buffering", "no")
                        self.send_header("Connection", "keep-alive")
                        self.end_headers()
                        while True:
                            chunk = resp.read(64)
                            if not chunk:
                                break
                            try:
                                self.wfile.write(chunk)
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError):
                                sys.stderr.write("[Model Load Stream] Client disconnected\n")
                                break
                        elapsed = time.perf_counter() - t0
                        sys.stderr.write(f"[Model Load Stream] Finished streaming load in {elapsed:.1f}s\n")
                        return

                    resp_data = resp.read()
                    elapsed = time.perf_counter() - t0
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp_data)))
                    self.end_headers()
                    self.wfile.write(resp_data)
                    sys.stderr.write(f"[Model Load] Loaded {payload.get('model')} in {elapsed:.1f}s\n")
                    return
            except urllib.error.HTTPError as e:
                err_data = e.read()
                elapsed = time.perf_counter() - t0
                sys.stderr.write(f"[Model Load Error] HTTP {e.code} in {elapsed:.1f}s: {err_data.decode('utf-8', errors='replace')[:120]}\n")
                self.send_response(e.code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_data)))
                self.end_headers()
                self.wfile.write(err_data)
                return
            except Exception as e:
                elapsed = time.perf_counter() - t0
                err_resp = json.dumps({"error": f"Model load failed after {elapsed:.1f}s: {str(e)}"}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_resp)))
                self.end_headers()
                self.wfile.write(err_resp)
                return

    def handle_health(self):
        """Query live /health on PS5 and return active model & runtime parameters."""
        target_ip = PS5_HOST
        target_port = 8081
        status_data = {
            "online": False,
            "target": DEFAULT_TARGET_URL,
            "ip": target_ip,
            "port": target_port,
            "model_loaded": False,
            "model_file": "unknown",
            "parameters": {}
        }

        # The PS5 serves one request at a time. While a job from this proxy is
        # running, a health request would only queue behind it and time out,
        # so answer from the last good reading instead.
        if INFERENCE_LOCK.locked():
            status_data.update(LAST_HEALTH)
            status_data["online"] = True
            status_data["busy"] = True
            resp_bytes = json.dumps(status_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)
            return

        try:
            req = urllib.request.Request(
                (PS5_BASE + "/health"),
                headers={"Authorization": f"Bearer {DEFAULT_API_KEY}"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                ps5_health = json.loads(resp.read().decode("utf-8"))
                status_data["online"] = True
                status_data["model_loaded"] = ps5_health.get("model_loaded", False)
                status_data["model_file"] = ps5_health.get("model_file", "unknown")
                status_data["image_model"] = ps5_health.get("image_model", None)
                status_data["stt_model"] = ps5_health.get("stt_model", None)
                status_data["tts_model"] = ps5_health.get("tts_model", None)
                status_data["parameters"] = ps5_health.get("parameters", {})

                img_m = status_data["image_model"] or ""
                stt_m = status_data["stt_model"] or ""
                tts_m = status_data["tts_model"] or ""

                status_data["voice_ready"] = bool(status_data["model_loaded"] and stt_m and tts_m)
                if "wan" in img_m.lower() or "video" in img_m.lower() or "hunyuan" in img_m.lower():
                    status_data["active_mode"] = "video"
                    status_data["active_model"] = img_m
                elif img_m:
                    status_data["active_mode"] = "image"
                    status_data["active_model"] = img_m
                elif stt_m:
                    status_data["active_mode"] = "stt"
                    status_data["active_model"] = stt_m
                elif tts_m:
                    status_data["active_mode"] = "tts"
                    status_data["active_model"] = tts_m
                elif status_data["model_loaded"] and status_data["model_file"] and status_data["model_file"] != "unknown":
                    status_data["active_mode"] = "llm"
                    status_data["active_model"] = status_data["model_file"]
                else:
                    status_data["active_mode"] = "idle"
                    status_data["active_model"] = "none"
                LAST_HEALTH.clear()
                LAST_HEALTH.update({k: v for k, v in status_data.items() if k not in ("online", "busy")})
        except Exception:
            # No answer in time: if the port still accepts connections, the PS5
            # is busy with someone else's request rather than offline.
            try:
                with socket.create_connection((target_ip, target_port), timeout=2):
                    status_data.update(LAST_HEALTH)
                    status_data["online"] = True
                    status_data["busy"] = True
            except Exception:
                status_data["online"] = False

        resp_bytes = json.dumps(status_data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def handle_list_image_models(self):
        """Returns the list of available Stable Diffusion models."""
        resp_bytes = json.dumps({"object": "list", "data": IMAGE_MODELS}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def handle_list_video_models(self):
        """Returns the list of available Video models."""
        resp_bytes = json.dumps({"object": "list", "data": VIDEO_MODELS}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def handle_list_stt_models(self):
        """Returns the list of available Whisper STT models."""
        resp_bytes = json.dumps({"object": "list", "data": STT_MODELS}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def handle_list_tts_models(self):
        """Returns the list of available TTS models."""
        resp_bytes = json.dumps({"object": "list", "data": TTS_MODELS}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def handle_list_all_audio_models(self):
        """Returns both STT and TTS models."""
        resp_bytes = json.dumps({
            "object": "list",
            "stt": STT_MODELS,
            "tts": TTS_MODELS
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def handle_image_generate(self):
        """Proxy text-to-image generation request to PS5 stable-diffusion.cpp endpoint."""
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            payload = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Invalid JSON body"}')
            return

        target_url = (PS5_BASE + "/v1/images/generations")
        is_stream = bool(payload.get("stream", True))
        payload["stream"] = is_stream

        # Sanitize payload for PS5 C++ backend:
        # C++ property_int strictly rejects negative values (v < 0). Passing seed = -1 causes HTTP 422.
        # When seed is omitted, the PS5 C++ backend uses its default (-1 random).
        if "seed" in payload:
            try:
                s = int(payload["seed"])
                if s < 0:
                    del payload["seed"]
                else:
                    payload["seed"] = s
            except (ValueError, TypeError):
                del payload["seed"]

        if "negative_prompt" in payload and not str(payload["negative_prompt"]).strip():
            del payload["negative_prompt"]

        # Ensure width and height are valid multiples of 64 (256..1024) and size string matches
        w = payload.get("width")
        h = payload.get("height")
        if w is not None and h is not None:
            try:
                wi = max(256, min(1024, (int(w) // 64) * 64))
                hi = max(256, min(1024, (int(h) // 64) * 64))
                payload["width"] = wi
                payload["height"] = hi
                payload["size"] = f"{wi}x{hi}"
            except (ValueError, TypeError):
                pass
        elif "size" in payload and "x" in str(payload["size"]).lower():
            try:
                parts = str(payload["size"]).lower().split("x")
                wi = max(256, min(1024, (int(parts[0]) // 64) * 64))
                hi = max(256, min(1024, (int(parts[1]) // 64) * 64))
                payload["width"] = wi
                payload["height"] = hi
                payload["size"] = f"{wi}x{hi}"
            except (ValueError, TypeError):
                pass

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEFAULT_API_KEY}"
        }
        if is_stream:
            headers["Accept"] = "text/event-stream"

        with INFERENCE_LOCK:
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(
                    target_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=300) as resp:
                    resp_content_type = resp.headers.get("Content-Type", "")
                    if is_stream or "text/event-stream" in resp_content_type:
                        self.send_response(resp.status)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("X-Accel-Buffering", "no")
                        self.send_header("Connection", "keep-alive")
                        self.end_headers()
                        while True:
                            line = resp.readline()
                            if not line:
                                break
                            try:
                                self.wfile.write(line)
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                                sys.stderr.write("[Image Gen Stream] Client disconnected\n")
                                self.close_connection = True
                                break
                            if b"data: [DONE]" in line:
                                break
                        elapsed = time.perf_counter() - t0
                        sys.stderr.write(f"[Image Gen Stream] Finished in {elapsed:.1f}s\n")
                        return

                    resp_data = resp.read()
                    elapsed = time.perf_counter() - t0
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp_data)))
                    self.end_headers()
                    try:
                        self.wfile.write(resp_data)
                    except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                        self.close_connection = True
                    sys.stderr.write(f"[Image Gen] Generated in {elapsed:.1f}s\n")
                    return
            except urllib.error.HTTPError as e:
                err_data = e.read()
                elapsed = time.perf_counter() - t0
                sys.stderr.write(f"[Image Gen Error] HTTP {e.code} in {elapsed:.1f}s: {err_data.decode('utf-8', errors='replace')[:120]}\n")
                try:
                    self.send_response(e.code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err_data)))
                    self.end_headers()
                    self.wfile.write(err_data)
                except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                    self.close_connection = True
                return
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError) as sock_err:
                sys.stderr.write(f"[Image Gen] Client disconnected: {sock_err}\n")
                self.close_connection = True
                return
            except Exception as e:
                elapsed = time.perf_counter() - t0
                err_resp = json.dumps({"error": f"Image generation failed after {elapsed:.1f}s: {str(e)}"}).encode("utf-8")
                try:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err_resp)))
                    self.end_headers()
                    self.wfile.write(err_resp)
                except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                    self.close_connection = True
                return

    def handle_video_generate(self):
        """Proxy text-to-video / image-to-video generation request to PS5 stable-diffusion.cpp endpoint."""
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            payload = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Invalid JSON body"}')
            return

        target_url = (PS5_BASE + "/v1/videos/generations")
        is_stream = bool(payload.get("stream", True))
        payload["stream"] = is_stream

        # Sanitize parameters for PS5 backend (omit negative seed to avoid 422)
        if "seed" in payload:
            try:
                s = int(payload["seed"])
                if s < 0:
                    del payload["seed"]
                else:
                    payload["seed"] = s
            except (ValueError, TypeError):
                del payload["seed"]

        if "negative_prompt" in payload and not str(payload["negative_prompt"]).strip():
            del payload["negative_prompt"]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEFAULT_API_KEY}"
        }
        if is_stream:
            headers["Accept"] = "text/event-stream"

        with INFERENCE_LOCK:
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(
                    target_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=3600) as resp:
                    resp_content_type = resp.headers.get("Content-Type", "")
                    if is_stream or "text/event-stream" in resp_content_type:
                        self.send_response(resp.status)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("X-Accel-Buffering", "no")
                        self.send_header("Connection", "keep-alive")
                        self.end_headers()
                        while True:
                            line = resp.readline()
                            if not line:
                                break
                            try:
                                self.wfile.write(line)
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                                sys.stderr.write("[Video Gen Stream] Client disconnected\n")
                                self.close_connection = True
                                break
                            if b"data: [DONE]" in line:
                                break
                        elapsed = time.perf_counter() - t0
                        sys.stderr.write(f"[Video Gen Stream] Finished in {elapsed:.1f}s\n")
                        return

                    resp_data = resp.read()
                    elapsed = time.perf_counter() - t0
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp_data)))
                    self.end_headers()
                    try:
                        self.wfile.write(resp_data)
                    except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                        self.close_connection = True
                    sys.stderr.write(f"[Video Gen] Generated in {elapsed:.1f}s\n")
                    return
            except urllib.error.HTTPError as e:
                err_data = e.read()
                elapsed = time.perf_counter() - t0
                sys.stderr.write(f"[Video Gen Error] HTTP {e.code} in {elapsed:.1f}s: {err_data.decode('utf-8', errors='replace')[:120]}\n")
                try:
                    self.send_response(e.code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err_data)))
                    self.end_headers()
                    self.wfile.write(err_data)
                except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                    self.close_connection = True
                return
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError) as sock_err:
                sys.stderr.write(f"[Video Gen] Client disconnected: {sock_err}\n")
                self.close_connection = True
                return
            except Exception as e:
                elapsed = time.perf_counter() - t0
                err_resp = json.dumps({"error": f"Video generation failed after {elapsed:.1f}s: {str(e)}"}).encode("utf-8")
                try:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err_resp)))
                    self.end_headers()
                    self.wfile.write(err_resp)
                except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                    self.close_connection = True
                return

    def handle_audio_transcribe(self):
        """Proxy speech-to-text request to PS5 whisper.cpp endpoint."""
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            payload = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Invalid JSON body"}')
            return

        target_url = (PS5_BASE + "/v1/audio/transcriptions")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEFAULT_API_KEY}"
        }

        with INFERENCE_LOCK:
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(
                    target_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    resp_data = resp.read()
                    elapsed = time.perf_counter() - t0
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp_data)))
                    self.end_headers()
                    try:
                        self.wfile.write(resp_data)
                    except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                        self.close_connection = True
                    sys.stderr.write(f"[STT Audio] Transcribed in {elapsed:.2f}s\n")
                    return
            except urllib.error.HTTPError as e:
                err_data = e.read()
                elapsed = time.perf_counter() - t0
                sys.stderr.write(f"[STT Error] HTTP {e.code} in {elapsed:.2f}s: {err_data.decode('utf-8', errors='replace')[:120]}\n")
                try:
                    self.send_response(e.code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err_data)))
                    self.end_headers()
                    self.wfile.write(err_data)
                except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                    self.close_connection = True
                return
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError) as sock_err:
                sys.stderr.write(f"[STT] Client disconnected: {sock_err}\n")
                self.close_connection = True
                return
            except Exception as e:
                elapsed = time.perf_counter() - t0
                err_resp = json.dumps({"error": f"Audio transcription failed after {elapsed:.2f}s: {str(e)}"}).encode("utf-8")
                try:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err_resp)))
                    self.end_headers()
                    self.wfile.write(err_resp)
                except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                    self.close_connection = True
                return

    def handle_audio_speech(self):
        """Proxy text-to-speech request to PS5 llama.cpp mtmd endpoint."""
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            payload = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Invalid JSON body"}')
            return

        target_url = (PS5_BASE + "/v1/audio/speech")

        # Sanitize parameters for PS5 backend (omit negative seed to avoid 422)
        if "seed" in payload:
            try:
                s = int(payload["seed"])
                if s < 0:
                    del payload["seed"]
                else:
                    payload["seed"] = s
            except (ValueError, TypeError):
                del payload["seed"]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEFAULT_API_KEY}"
        }

        with INFERENCE_LOCK:
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(
                    target_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    resp_data = resp.read()
                    content_type = resp.headers.get("Content-Type", "audio/wav")
                    elapsed = time.perf_counter() - t0
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(resp_data)))
                    self.end_headers()
                    try:
                        self.wfile.write(resp_data)
                        self.wfile.flush()
                        sys.stderr.write(f"[TTS Audio] Synthesized in {elapsed:.2f}s ({len(resp_data)} bytes)\n")
                    except (BrokenPipeError, ConnectionResetError, ssl.SSLError) as write_err:
                        sys.stderr.write(f"[TTS Audio] Client disconnected during audio send: {write_err}\n")
                        self.close_connection = True
                    return
            except urllib.error.HTTPError as e:
                err_data = e.read()
                elapsed = time.perf_counter() - t0
                sys.stderr.write(f"[TTS Error] HTTP {e.code} in {elapsed:.2f}s: {err_data.decode('utf-8', errors='replace')[:120]}\n")
                try:
                    self.send_response(e.code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err_data)))
                    self.end_headers()
                    self.wfile.write(err_data)
                except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                    self.close_connection = True
                return
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError) as sock_err:
                sys.stderr.write(f"[TTS Audio] Client disconnected: {sock_err}\n")
                self.close_connection = True
                return
            except Exception as e:
                elapsed = time.perf_counter() - t0
                err_resp = json.dumps({"error": f"Speech synthesis failed after {elapsed:.2f}s: {str(e)}"}).encode("utf-8")
                try:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err_resp)))
                    self.end_headers()
                    self.wfile.write(err_resp)
                except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                    self.close_connection = True
                return

    def handle_voice_chat_proxy(self):
        """Proxy real-time voice agent SSE request to PS5 /v1/voice/chat endpoint."""
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            payload = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Invalid JSON body"}')
            return

        target_url = (PS5_BASE + "/v1/voice/chat")
        audio_b64 = str(payload.get("audio_b64", "")).strip()
        if not audio_b64 or len(audio_b64) < 50:
            self.send_response(422)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"audio_b64 (base64 WAV) is required; please hold or tap and speak clearly."}}')
            return
        payload["audio_b64"] = audio_b64

        # Strip empty messages array (PS5 C++ backend rejects empty messages array roles.empty() with HTTP 422)
        if "messages" in payload:
            if not isinstance(payload["messages"], list) or len(payload["messages"]) == 0:
                del payload["messages"]
            else:
                clean_msgs = []
                for m in payload["messages"]:
                    if isinstance(m, dict) and m.get("role") in ["user", "assistant", "system"] and str(m.get("content", "")).strip():
                        clean_msgs.append({"role": m["role"], "content": str(m["content"]).strip()})
                if clean_msgs:
                    payload["messages"] = clean_msgs
                else:
                    del payload["messages"]

        if "speaker_b64" in payload and (not payload["speaker_b64"] or not str(payload["speaker_b64"]).strip()):
            del payload["speaker_b64"]

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {DEFAULT_API_KEY}"
        }

        with INFERENCE_LOCK:
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(
                    target_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=300) as resp:
                    self.send_response(resp.status)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Accel-Buffering", "no")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    while True:
                        line = resp.readline()
                        if not line:
                            break
                        try:
                            self.wfile.write(line)
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                            sys.stderr.write("[Voice Chat Stream] Client disconnected\n")
                            self.close_connection = True
                            break
                        if b"data: [DONE]" in line:
                            break
                    elapsed = time.perf_counter() - t0
                    sys.stderr.write(f"[Voice Chat Stream] Completed in {elapsed:.1f}s\n")
                    return
            except urllib.error.HTTPError as e:
                err_data = e.read()
                elapsed = time.perf_counter() - t0
                sys.stderr.write(f"[Voice Chat Error] HTTP {e.code} in {elapsed:.1f}s: {err_data.decode('utf-8', errors='replace')[:120]}\n")
                try:
                    self.send_response(e.code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err_data)))
                    self.end_headers()
                    self.wfile.write(err_data)
                except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                    self.close_connection = True
                return
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError) as sock_err:
                sys.stderr.write(f"[Voice Chat] Client disconnected: {sock_err}\n")
                self.close_connection = True
                return
            except Exception as e:
                elapsed = time.perf_counter() - t0
                err_resp = json.dumps({"error": f"Voice chat failed after {elapsed:.1f}s: {str(e)}"}).encode("utf-8")
                try:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err_resp)))
                    self.end_headers()
                    self.wfile.write(err_resp)
                except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                    self.close_connection = True
                return

    def handle_list_tools(self):
        """Returns the list of available tools."""
        tools_list = list(AVAILABLE_TOOLS.keys())
        resp_bytes = json.dumps({"tools": tools_list}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def handle_execute_tool(self):
        """Execute a local tool function."""
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        t_start = time.perf_counter()

        try:
            payload = json.loads(post_data.decode("utf-8"))
            tool_name = payload.get("name")
            tool_args = payload.get("arguments", {})

            if tool_name not in AVAILABLE_TOOLS:
                raise ValueError(f"Unknown tool: '{tool_name}'")

            fn = AVAILABLE_TOOLS[tool_name]
            if isinstance(tool_args, dict):
                result = fn(**tool_args)
            elif isinstance(tool_args, (list, tuple)):
                result = fn(*tool_args)
            else:
                result = fn(tool_args)

            elapsed_ms = (time.perf_counter() - t_start) * 1000
            response_obj = {
                "success": True,
                "name": tool_name,
                "result": result,
                "execution_time_ms": round(elapsed_ms, 2)
            }
            status_code = 200

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            response_obj = {
                "success": False,
                "name": payload.get("name") if 'payload' in locals() else "unknown",
                "error": str(e),
                "execution_time_ms": round(elapsed_ms, 2)
            }
            status_code = 400

        resp_bytes = json.dumps(response_obj).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def handle_chat_proxy(self):
        """Proxy streaming chat completion request to the PS5 endpoint with payload sanitization."""
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        # Sanitize payload for PS5 C++ backend (strip oversized tool schemas, format messages)
        is_stream = True
        try:
            payload = json.loads(post_data.decode("utf-8"))
            messages = payload.get("messages", [])
            clean_messages = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, list):
                    text_parts = []
                    image_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, dict) and part.get("type") == "image_url":
                            image_parts.append(part)
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content = "\n".join(text_parts)
                    # The PS5 takes images as OpenAI content parts (models with a vision projector).
                    if image_parts and role == "user":
                        content = image_parts + [{"type": "text", "text": content}]
                elif not isinstance(content, str):
                    content = str(content)

                if role in ["system", "user", "assistant"]:
                    clean_messages.append({"role": role, "content": content})
                elif role == "tool":
                    clean_messages.append({"role": "user", "content": f"[Tool Output]: {content}"})

            is_stream = payload.get("stream", True)
            clean_payload = {
                "model": payload.get("model", "ps5-local"),
                "messages": clean_messages,
                "max_tokens": min(payload.get("max_tokens", 1024), 4096),
                "temperature": payload.get("temperature", 0.7),
                "top_p": payload.get("top_p", 1.0),
                "repeat_penalty": payload.get("repeat_penalty", 1.15),
                "stream": is_stream
            }
            forward_data = json.dumps(clean_payload).encode("utf-8")
        except Exception:
            forward_data = post_data

        target_url = self.headers.get("X-Target-Endpoint", DEFAULT_TARGET_URL)
        auth_header = self.headers.get("Authorization", f"Bearer {DEFAULT_API_KEY}")
        req = urllib.request.Request(
            target_url,
            data=forward_data,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if is_stream else "application/json",
                "Authorization": auth_header
            },
            method="POST"
        )

        try:
            with INFERENCE_LOCK:
                with urllib.request.urlopen(req, timeout=180) as target_resp:
                    if is_stream:
                        self.send_response(target_resp.status)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("X-Accel-Buffering", "no")
                        self.send_header("Connection", "keep-alive")
                        self.end_headers()

                        # Stream response line by line directly to the client
                        while True:
                            line = target_resp.readline()
                            if not line:
                                break
                            try:
                                self.wfile.write(line)
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                                sys.stderr.write("[Chat Stream] Client disconnected\n")
                                self.close_connection = True
                                break
                            if b"data: [DONE]" in line:
                                break
                    else:
                        resp_data = target_resp.read()
                        self.send_response(target_resp.status)
                        self.send_header("Content-Type", "application/json; charset=utf-8")
                        self.send_header("Content-Length", str(len(resp_data)))
                        self.end_headers()
                        try:
                            self.wfile.write(resp_data)
                        except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                            self.close_connection = True

        except urllib.error.HTTPError as e:
            error_body = e.read()
            print(f"[ERROR] PS5 HTTP error {e.code}: {error_body.decode('utf-8', errors='ignore')}", file=sys.stderr)
            try:
                self.send_response(e.code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(error_body)))
                self.end_headers()
                self.wfile.write(error_body)
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                self.close_connection = True
        except (BrokenPipeError, ConnectionResetError, ssl.SSLError) as sock_err:
            sys.stderr.write(f"[Chat Proxy] Client disconnected: {sock_err}\n")
            self.close_connection = True
        except Exception as e:
            print(f"[ERROR] Proxy exception: {str(e)}", file=sys.stderr)
            try:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                error_msg = json.dumps({"error": f"Failed to reach PS5 endpoint: {str(e)}"}).encode("utf-8")
                self.send_header("Content-Length", str(len(error_msg)))
                self.end_headers()
                self.wfile.write(error_msg)
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                self.close_connection = True

def get_tailscale_ip():
    try:
        proc = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=1)
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return None

def start_https_server(port=8443):
    cert_path = os.path.join(STATIC_DIR, "certs", os.environ.get("PSLLAMA_TLS_NAME", "server") + ".crt")
    key_path = os.path.join(STATIC_DIR, "certs", os.environ.get("PSLLAMA_TLS_NAME", "server") + ".key")
    if not (os.path.exists(cert_path) and os.path.exists(key_path)):
        return None
    try:
        import ssl
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
        httpsd = ThreadingHTTPServer(("", port), ChatProxyHandler)
        httpsd.socket = ctx.wrap_socket(httpsd.socket, server_side=True)
        t = threading.Thread(target=httpsd.serve_forever, daemon=True)
        t.start()
        return httpsd
    except Exception as e:
        sys.stderr.write(f"[HTTPS Notice] Could not start HTTPS listener on {port}: {e}\n")
        return None

def run(port=PORT):
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, ChatProxyHandler)
    https_server = start_https_server(port=8443)
    ts_ip = get_tailscale_ip()
    print("=" * 60)
    print(f"🚀 PS5 Local LLM Chat UI active:")
    print(f"   • Local:     http://localhost:{port}")
    if ts_ip:
        print(f"   • Tailscale: http://{ts_ip}:{port}")
        print(f"   • MagicDNS:  http://host.example.ts.net:{port}")
    if https_server:
        print(f"   • HTTPS:     https://host.example.ts.net:8443")
    print(f"🔗 Target Endpoint: {DEFAULT_TARGET_URL}")
    print(f"📁 Serving static directory: {STATIC_DIR}")
    print(f"🛠️  Available Tools: {list(AVAILABLE_TOOLS.keys())}")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down gracefully.")
        httpd.server_close()
        if https_server:
            https_server.server_close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PS5 Local LLM Chat Proxy Server")
    parser.add_argument("--port", type=int, default=PORT, help=f"Server port (default: {PORT})")
    args = parser.parse_args()
    run(args.port)

