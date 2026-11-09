"""Materialize the pinned checkpoint from the host's secret file and start Inksy."""

import base64
import binascii
import hashlib
import os
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    source = Path(os.environ.get("INKSY_MODEL_SECRET_FILE", "/etc/secrets/inksy-model.b64"))
    expected = os.environ.get("INKSY_CHECKPOINT_SHA256", "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise SystemExit("Set INKSY_CHECKPOINT_SHA256 to the intended checkpoint SHA-256.")
    try:
        with source.open("rb") as secret:
            encoded = secret.read(1_000_001)
        if len(encoded) > 1_000_000:
            raise SystemExit("The model secret file exceeds 1 MB.")
        model = base64.b64decode(b"".join(encoded.split()), validate=True)
    except (OSError, binascii.Error):
        raise SystemExit("Install inksy-model.b64 as a valid base64 Render secret file.") from None
    if hashlib.sha256(model).hexdigest() != expected:
        raise SystemExit("Checkpoint SHA-256 does not match the configured deployment model.")

    with TemporaryDirectory(prefix="inksy-model-") as directory:
        checkpoint = Path(directory) / "model.pt"
        checkpoint.write_bytes(model)
        sys.argv[1:] = [
            "--host",
            "0.0.0.0",
            "--hosted",
            "--checkpoint",
            str(checkpoint),
            "--style-dir",
            os.environ.get("INKSY_STYLE_DIR", "/tmp/inksy-personal-styles"),
        ]
        from rendering.studio import main as serve_studio

        serve_studio()


if __name__ == "__main__":
    main()
