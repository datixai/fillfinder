"""Local test server. Run:  python local_server.py   then open http://localhost:8000"""
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))
from index import run_request, MAX_UPLOAD  # noqa: E402

PUBLIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=PUBLIC, **kw)

    def do_POST(self):
        import json
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD * 2:
            status, obj = 413, {"ok": False, "error": "File is too large (limit 12 MB)."}
        else:
            status, obj = run_request(self.rfile.read(length))
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print("FillFinder running at http://localhost:%d  (Ctrl+C to stop)" % port)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
