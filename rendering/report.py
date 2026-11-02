"""Portable, escaped HTML report with source text, model provenance and actual PNG downloads."""

# Long literal lines preserve readable HTML/CSS in the self-contained report template.
# ruff: noqa: E501

import html
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def preview_html(manifest: dict) -> str:
    source = html.escape(manifest["original_text"])
    normalized = html.escape(manifest["rendered_text"])
    normalization = (
        "<details open><summary>Normalized text used for generation</summary><pre>"
        + normalized
        + "</pre></details>"
        if manifest["normalization_applied"]
        else ""
    )
    page_cards = []
    for page in manifest["pages"]:
        filename = html.escape(page["filename"], quote=True)
        number = page["index"]
        page_cards.append(
            f'<article class="page"><div class="page-title"><span>Page {number} '
            f"<small>{page['width']} x {page['height']} px</small></span>"
            f'<a href="{filename}" download>Download PNG</a></div>'
            f'<div class="canvas"><img src="{filename}" alt="Generated handwriting, page {number}" '
            'loading="lazy"></div></article>'
        )
    warnings = "".join(f"<li>{html.escape(message)}</li>" for message in manifest["warnings"])
    options = manifest["layout"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inksy - Local handwriting preview</title>
<style>
:root{{font-family:"Segoe UI",Arial,sans-serif;color:#17283e;background:#eff3f7}}
*{{box-sizing:border-box}}body{{margin:0}}a{{color:#087d80;text-underline-offset:3px}}
header{{background:#fff;border-bottom:1px solid #d9e2eb;padding:24px 32px}}
.head{{max-width:1440px;margin:auto;display:flex;align-items:center;gap:16px;flex-wrap:wrap}}
h1{{font-size:27px;margin:0;letter-spacing:3px}}.tag{{background:#e7f3f2;color:#106b6e;padding:5px 10px;border-radius:20px;font-size:12px}}
.subtitle{{color:#647388;font-size:14px;margin:8px 0 0}}
main{{max-width:1440px;margin:24px auto;padding:0 24px;display:grid;grid-template-columns:340px minmax(0,1fr);gap:24px;align-items:start}}
aside{{background:white;border:1px solid #dce4eb;border-radius:12px;padding:22px;position:sticky;top:20px}}
h2{{font-size:17px;margin:0 0 16px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.8 "Segoe UI",Arial,sans-serif;margin:0}}
.metadata{{border-top:1px solid #e4eaf0;margin-top:20px;padding-top:14px;color:#596a80;font-size:12px;line-height:1.8}}
details{{margin-top:16px}}summary{{cursor:pointer}}ul{{padding-left:17px}}li{{margin-bottom:7px}}
.toolbar{{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:16px;font-size:13px}}
.button{{display:inline-block;background:#137f82;color:white;border-radius:7px;padding:9px 13px;text-decoration:none}}
.page{{background:#fff;border:1px solid #dce4eb;border-radius:10px;margin-bottom:22px;overflow:hidden}}
.page-title{{display:flex;justify-content:space-between;gap:10px;padding:13px 18px;font-size:13px;border-bottom:1px solid #e4eaf0}}
small{{color:#738095;margin-left:8px}}.canvas{{overflow:auto;padding:20px;background:#e5ebf1;text-align:center}}
.canvas img{{width:100%;height:auto;display:block;margin:auto;box-shadow:0 3px 15px #17283e1c}}
input[type=range]{{vertical-align:middle;width:130px}}code{{font-size:11px;overflow-wrap:anywhere}}
footer{{max-width:1440px;margin:0 auto;padding:0 24px 28px;color:#647388;font-size:12px}}
@media(max-width:850px){{main{{grid-template-columns:1fr}}aside{{position:static}}header{{padding:20px 24px}}.canvas{{padding:10px}}}}
</style></head><body>
<header><div class="head"><h1>INKSY</h1><span class="tag">Experimental local preview</span></div>
<div class="head"><p class="subtitle">Your text, rendered with our trained character model.</p></div></header>
<main><aside><h2>Original typed text</h2><pre>{source}</pre>{normalization}
<div class="metadata">Seed: <b>{manifest["seed"]}</b><br>Model glyphs: {manifest["coverage"]["model_glyphs"]}<br>
Procedural punctuation: {manifest["coverage"]["procedural_punctuation"]}<br>
Glyph size: {options["glyph_size"]} px; character gap: {options["character_spacing"]} px<br>
Line gap: {options["line_spacing"]} px; paper: {html.escape(options["paper"])}<br>
Generation + rendering: {manifest["duration_seconds"]:.2f} seconds
<details><summary>Model and rendering details</summary><p>Checkpoint: <code>{html.escape(manifest["checkpoint"])}</code></p>
<p>SHA-256: <code>{manifest["checkpoint_sha256"]}</code></p><p>Renderer: {manifest["renderer_version"]}</p></details>
<details open><summary>What this preview means</summary><ul>{warnings}</ul></details></div></aside>
<section aria-label="Generated handwriting"><div class="toolbar"><span>{manifest["page_count"]} generated page(s)</span>
<label>Display zoom <input id="zoom" type="range" min="40" max="160" value="100" aria-label="Preview zoom">
<output id="zoom-label">100%</output></label><a class="button" href="pages.zip" download>Download all PNGs</a></div>
{"".join(page_cards)}</section></main><footer>Local report. Zoom changes the display only. Rerun the CLI with new text or layout settings to generate a new preview.</footer>
<script>document.getElementById('zoom').addEventListener('input',function(){{
document.getElementById('zoom-label').textContent=this.value+'%';
document.querySelectorAll('.canvas img').forEach(img=>img.style.width=this.value+'%');
}});</script></body></html>"""


def write_report(output: Path, manifest: dict):
    (output / "preview.html").write_bytes(preview_html(manifest).encode("utf-8"))
    with ZipFile(output / "pages.zip", "w", compression=ZIP_DEFLATED) as archive:
        for page in manifest["pages"]:
            archive.write(output / page["filename"], arcname=page["filename"])
