"""Construit le notebook autonome à partir du script pédagogique à cellules."""
from pathlib import Path
import re
import nbformat

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "scripts" / "tp2_source.py").read_text(encoding="utf-8")
parts = re.split(r"^# %%(.*)\n", source, flags=re.MULTILINE)
cells = []
for index in range(1, len(parts), 2):
    marker, body = parts[index], parts[index + 1].strip("\n")
    if "[markdown]" in marker:
        lines = [line[2:] if line.startswith("# ") else line[1:] if line == "#" else line
                 for line in body.splitlines()]
        cell = nbformat.v4.new_markdown_cell("\n".join(lines))
        if "[resume]" in marker:
            cell.metadata["tags"] = ["resume-resultats"]
    else:
        cell = nbformat.v4.new_code_cell(body)
    cells.append(cell)
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3.11 — TP2 MNIST", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
})
nbformat.validate(notebook)
target = ROOT / "TP2_Modeles_generatifs_MNIST.ipynb"
nbformat.write(notebook, target)
print(f"Notebook créé : {target.name} ({len(cells)} cellules)")
