"""Build the project guide PDF: stitch the chapters in src/ together and print them with Edge/Chrome.

    python docs/guide/make_figures.py     # (re)compute the figures, once
    python docs/guide/build.py            # -> docs/RescueThermal_AI_Guide.pdf
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_HTML = HERE / "guide.html"
OUT_PDF = HERE.parent / "RescueThermal_AI_Guide.pdf"
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "google-chrome", "chromium", "msedge"]


def browser() -> str:
    for b in BROWSERS:
        if Path(b).exists() or shutil.which(b):
            return b
    sys.exit("No Edge or Chrome found: open docs/guide/guide.html in a browser and print it to PDF.")


def main():
    parts = sorted((HERE / "src").glob("*.html"))
    OUT_HTML.write_text("\n".join(p.read_text(encoding="utf-8") for p in parts) + "\n</body>\n</html>\n", encoding="utf-8")
    profile = HERE / ".browser-profile"
    subprocess.run([browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--user-data-dir={profile}", "--virtual-time-budget=20000",
                    f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()], check=True, capture_output=True, timeout=300)
    shutil.rmtree(profile, ignore_errors=True)
    pages = len(re.findall(rb"/Type\s*/Page[^s]", OUT_PDF.read_bytes()))
    print(f"{OUT_PDF}: {pages} pages")


if __name__ == "__main__":
    main()
