"""Serve the Inksy handwriting studio on the local machine."""

import argparse
import base64
import io
import json
import os
import secrets
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore
from urllib.parse import urlsplit

import torch

from ml.characters import ALPHABET
from ml.print_handwriting import load_letter_variants, with_print_punctuation
from rendering.personal_styles import PersonalStyles
from rendering.print_pages import MAX_TEXT, print_pages
from rendering.worksheet_photo import photo_samples, template

WEB = Path(__file__).with_name("studio_web")
STATIC = {
    "/": ("index.html", "text/html"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
    "/studio.css": ("studio.css", "text/css"),
    "/studio.js": ("studio.js", "text/javascript"),
    "/train": ("train.html", "text/html"),
    "/train.js": ("train.js", "text/javascript"),
    "/worksheet": ("worksheet.html", "text/html"),
}
DEFAULT_CHECKPOINT = Path("artifacts/print-handwriting-v3-balanced/model.pt")
PRIVATE_GET = {"/train", "/train.js", "/worksheet", "/worksheet.png", "/api/training"}
PRIVATE_POST = {"/api/train", "/api/worksheet-photo"}


def encode_page(page):
    stream = io.BytesIO()
    page.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def handler_for(bank, styles, *, hosted=False, admin_password=None):
    admin_header = (
        b"Basic " + base64.b64encode(f"inksy:{admin_password}".encode()) if admin_password else b""
    )
    session_token = secrets.token_urlsafe(32)
    render_slots = BoundedSemaphore(2)

    class StudioHandler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            if hosted:
                self.connection.settimeout(20)

        def is_admin(self):
            if not hosted:
                return True
            if not admin_header:
                return False
            try:
                cookie = SimpleCookie(self.headers.get("Cookie", ""))
            except CookieError:
                return False
            entry = cookie.get("inksy_admin")
            return entry is not None and secrets.compare_digest(
                entry.value.encode("utf-8"), session_token.encode("ascii")
            )

        def send_content(self, status, content_type, body, headers=()):
            self.send_response(status)
            self.send_header(
                "Content-Type",
                content_type
                if content_type.startswith("image/")
                else f"{content_type}; charset=utf-8",
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in headers:
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status, result):
            self.send_content(status, "application/json", json.dumps(result).encode("utf-8"))

        def do_GET(self):
            route = urlsplit(self.path).path
            headers = ()
            if route in PRIVATE_GET and not self.is_admin():
                if route != "/train" or not admin_header:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                supplied = self.headers.get("Authorization", "").encode("utf-8")
                if not secrets.compare_digest(supplied, admin_header):
                    self.send_content(
                        HTTPStatus.UNAUTHORIZED,
                        "text/plain",
                        b"Sign in to train a style.",
                        [("WWW-Authenticate", 'Basic realm="Inksy training", charset="UTF-8"')],
                    )
                    return
                headers = [
                    (
                        "Set-Cookie",
                        f"inksy_admin={session_token}; Path=/; Secure; HttpOnly; SameSite=Strict",
                    )
                ]
            if route == "/worksheet.png":
                self.send_content(HTTPStatus.OK, "image/png", template())
                return
            if route == "/api/status":
                self.send_json(HTTPStatus.OK, {"ready": True, "max_text": MAX_TEXT})
                return
            if route == "/api/styles":
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "styles": styles.list() if self.is_admin() else [],
                        "hosted": hosted,
                        "training_available": not hosted or bool(admin_header),
                    },
                )
                return
            if route == "/api/training":
                self.send_json(HTTPStatus.OK, styles.status())
                return
            if route not in STATIC:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            name, content_type = STATIC[route]
            self.send_content(HTTPStatus.OK, content_type, (WEB / name).read_bytes(), headers)

        def do_POST(self):
            route = urlsplit(self.path).path
            if route != "/api/render" and (route not in PRIVATE_POST or not self.is_admin()):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if self.headers.get_content_type() != "application/json":
                self.send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Send JSON text."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                maximum = 6_300_000 if route != "/api/render" else 65_536
                if not 0 < length <= maximum:
                    raise ValueError("The request is empty or too large.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Send a JSON object.")
                if route == "/api/worksheet-photo":
                    samples = photo_samples(payload.get("photo"), payload.get("markers"))
                    self.send_json(HTTPStatus.OK, {"samples": samples})
                    return
                if route == "/api/train":
                    styles.start(
                        payload.get("name"),
                        payload.get("samples"),
                        full=payload.get("alphabet") == "alphanumeric",
                        photo=payload.get("photoImported") is True,
                    )
                    self.send_json(HTTPStatus.ACCEPTED, styles.status())
                    return
                if not isinstance(payload.get("text"), str):
                    raise ValueError("Enter text to generate handwriting.")
                style = payload.get("style", "standard")
                if style != "standard" and not self.is_admin():
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "That style is unavailable."})
                    return
                if hosted and not render_slots.acquire(blocking=False):
                    self.send_json(
                        HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Studio busy. Try again shortly."}
                    )
                    return
                try:
                    pages = print_pages(
                        payload["text"], bank if style == "standard" else styles.bank(style)
                    )
                    result = {"pages": [encode_page(page) for page in pages]}
                finally:
                    if hosted:
                        render_slots.release()
            except (UnicodeError, ValueError) as error:
                message = "Invalid JSON." if isinstance(error, json.JSONDecodeError) else str(error)
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": message})
                return
            self.send_json(HTTPStatus.OK, result)

    return StudioHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=os.environ.get("PORT", "8765"))
    parser.add_argument("--hosted", action="store_true", help="Enable public demo access rules.")
    parser.add_argument("--style-dir", type=Path, default=Path("artifacts/personal-styles"))
    args = parser.parse_args()
    torch.set_num_threads(1 if args.hosted else 4)
    bank, _ = load_letter_variants(args.checkpoint)
    if set(bank) != set(ALPHABET):
        parser.error("The studio requires a mixed-case print checkpoint.")
    handler = handler_for(
        with_print_punctuation(bank),
        PersonalStyles(args.style_dir),
        hosted=args.hosted,
        admin_password=os.environ.get("INKSY_ADMIN_PASSWORD"),
    )
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f"Listening on {args.host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
