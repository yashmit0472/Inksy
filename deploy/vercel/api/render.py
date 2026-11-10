"""Render pages from the standard model's exported glyphs on Vercel."""

import base64
import gzip
import io
import json
from functools import cache
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from PIL import Image

from rendering.print_pages import MAX_TEXT, print_pages


@cache
def glyph_bank():
    source = Path(__file__).resolve().parents[1] / "data" / "glyphs.json.gz"
    with gzip.open(source, "rt", encoding="utf-8") as file:
        rows = json.load(file)
    return {
        char: [Image.frombytes("L", (width, 48), base64.b64decode(raw)) for width, raw in tiles]
        for char, tiles in rows.items()
    }


class handler(BaseHTTPRequestHandler):
    def send_json(self, status, value):
        body = json.dumps(value, separators=(",", ":")).encode()
        if len(body) > 4_400_000:
            status, body = 400, b'{"error":"This preview is too large. Shorten your text."}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.send_json(200, {"ready": bool(glyph_bank()), "max_text": MAX_TEXT})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65_536 or self.headers.get_content_type() != "application/json":
                raise ValueError("Send a JSON request with text.")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict) or not isinstance(data.get("text"), str):
                raise ValueError("Enter text to generate handwriting.")
            if data.get("style", "standard") != "standard":
                self.send_json(404, {"error": "That style is unavailable."})
                return
            pages = []
            for page in print_pages(data["text"], glyph_bank()):
                with io.BytesIO() as stream:
                    page.save(stream, format="PNG")
                    pages.append(base64.b64encode(stream.getvalue()).decode("ascii"))
        except (UnicodeError, ValueError) as error:
            self.send_json(
                400,
                {
                    "error": "Invalid JSON."
                    if isinstance(error, json.JSONDecodeError)
                    else str(error)
                },
            )
            return
        self.send_json(200, {"pages": pages})
