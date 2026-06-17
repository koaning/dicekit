from marimo._ast.app import InternalApp
from __init__ import app
from pathlib import Path

# Lets move out the code first

cells = InternalApp(app).graph.cells
codes = [v.code for v in cells.values() if v.language=="python" and "## Export" in v.code]

code_export = ""
for code in codes:
    code_export += code.replace("## Export", "") + "\n"

Path("dicekit/__init__.py").write_text(code_export)
