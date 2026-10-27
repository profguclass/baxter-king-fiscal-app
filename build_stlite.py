"""Build a static, server-free copy of the app into docs/ using stlite
(Streamlit running in the browser via Pyodide). Every student's browser runs
the Python itself, so the host only serves files and load does not grow with
the number of users.

    python build_stlite.py
    python -m http.server 8000 --directory docs     # preview locally

Publish by serving docs/ (e.g. GitHub Pages: Settings > Pages > main /docs).
app.py and fiscal_model.py stay the single source of truth; docs/ is generated.
"""
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "docs"
FILES = ["app.py", "appendix.py", "fiscal_model.py", "kgu_emblem.png"]
STLITE = "0.75.0"

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Fiscal Policy in General Equilibrium</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/mountable@{v}/build/stlite.css" />
</head>
<body>
  <div id="root">Loading the model (first visit downloads the Python runtime, ~20-40 MB)...</div>
  <script src="https://cdn.jsdelivr.net/npm/@stlite/mountable@{v}/build/stlite.js"></script>
  <script>
    stlite.mount({{
      requirements: ["numpy", "pandas", "plotly"],
      entrypoint: "app.py",
      files: {{
{files}
      }},
    }}, document.getElementById("root"));
  </script>
</body>
</html>
"""

OUT.mkdir(exist_ok=True)
for name in FILES:
    shutil.copy(ROOT / name, OUT / name)
# A content hash in each URL makes browsers fetch a file again as soon as it
# changes, instead of reusing GitHub Pages' cached copy (up to 10 minutes).
entries = ",\n".join(
    f'        "{n}": {{ url: "./{n}?v={hashlib.sha256((OUT / n).read_bytes()).hexdigest()[:10]}" }}'
    for n in FILES
)
(OUT / "index.html").write_text(HTML.format(v=STLITE, files=entries), encoding="utf-8")
print(f"Wrote {OUT / 'index.html'} (+ {', '.join(FILES)})")
