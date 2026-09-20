"""Vercel Python entrypoint (api/index.py). POST /api/process

Body (JSON):
  {"filename": "design.dst", "data": "<base64 of the file>", "options": {...}}
Response (JSON):
  {"ok": true, "steps": [...], "stats": {...}, "downloads": {"name": "<base64>"}}
"""
import json
import os
from http.server import BaseHTTPRequestHandler

try:  # works both on Vercel (api/ as package root) and locally
    from fillfinder_core import process
except ImportError:  # pragma: no cover
    from .fillfinder_core import process

MAX_UPLOAD = 12 * 1024 * 1024  # 12 MB


def run_request(body_bytes):
    """Shared by the Vercel handler and the local dev server."""
    try:
        payload = json.loads(body_bytes.decode("utf-8"))
    except Exception:
        return 400, {"ok": False, "error": "Could not read the request."}
    name = payload.get("filename") or "design.dst"
    data = payload.get("data") or ""
    opts = payload.get("options") or {}
    import base64
    try:
        raw = base64.b64decode(data)
    except Exception:
        return 400, {"ok": False, "error": "The uploaded file could not be decoded."}
    if not raw:
        return 400, {"ok": False, "error": "No file was uploaded."}
    if len(raw) > MAX_UPLOAD:
        return 413, {"ok": False, "error": "File is too large (limit 12 MB)."}
    clean = {}
    for k, cast in (("thick", float), ("gap", float), ("overlap", float), ("min_area", float),
                    ("max_area", float), ("angle", float), ("spacing", float),
                    ("fill_all", bool), ("make_stitches", bool)):
        if k in opts and opts[k] is not None:
            try:
                clean[k] = cast(opts[k])
            except (TypeError, ValueError):
                pass
    try:
        return 200, process(raw, name, clean)
    except ValueError as e:
        return 400, {"ok": False, "error": str(e)}
    except MemoryError:
        return 500, {"ok": False, "error": "This design is too heavy to process here. Try the desktop version."}
    except Exception as e:  # pragma: no cover
        return 500, {"ok": False, "error": "Something went wrong: %s" % e}


class handler(BaseHTTPRequestHandler):
    def _send(self, status, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/").endswith("/api/process") or self.path.startswith("/api/"):
            self._send(200, {"ok": True, "message": "FillFinder API is running. Send a POST request."})
            return
        page = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public", "index.html")
        try:
            with open(page, "rb") as f:
                body = f.read()
        except OSError:
            self._send(404, {"ok": False, "error": "Page not found."})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD * 2:
            self._send(413, {"ok": False, "error": "File is too large (limit 12 MB)."})
            return
        status, obj = run_request(self.rfile.read(length))
        self._send(status, obj)

    def log_message(self, *args):  # keep the Vercel logs clean
        pass
