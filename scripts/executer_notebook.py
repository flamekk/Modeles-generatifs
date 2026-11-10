"""Exécute le TP de haut en bas, conserve les sorties et prépare une prévisualisation."""
from pathlib import Path
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in ("jupyter", "ipython", "matplotlib"):
    (ROOT / ".qa" / directory).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".qa" / "jupyter"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".qa" / "ipython"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".qa" / "matplotlib"))

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

os.chdir(ROOT)
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
target = ROOT / "TP2_Modeles_generatifs_MNIST.ipynb"
notebook = nbformat.read(target, as_version=4)
started = time.perf_counter()

def report_cell(cell, cell_index, **kwargs):
    if cell.cell_type == "code":
        print(f"Cellule {cell_index + 1}/{len(notebook.cells)} : {cell.source.splitlines()[0][:100]}", flush=True)

def report_outputs(cell, **kwargs):
    streams = "".join(output.get("text", "") for output in cell.get("outputs", []) if output.get("output_type") == "stream")
    if streams:
        print(streams[-2400:], flush=True)

client = NotebookClient(notebook, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}, on_cell_start=report_cell, on_cell_executed=report_outputs)
try:
    client.execute()
except Exception:
    nbformat.write(notebook, target)
    print("Exécution interrompue : sorties partielles conservées.", flush=True)
    raise

metrics = json.loads((ROOT / "resultats" / "metriques.json").read_text(encoding="utf-8"))
classifier, ae, vae = metrics["classifieur"], metrics["autoencodeur"], metrics["vae"]
for cell in notebook.cells:
    if "resume-resultats" in cell.metadata.get("tags", []):
        cell.source = (
            "## Résultats de l'exécution de référence\n\n"
            f"Exécution complète sur **{metrics['configuration']['n_train']:,} images d'entraînement**, "
            f"**{metrics['configuration']['n_validation']:,} de validation** et **{metrics['configuration']['n_test']:,} de test**, "
            "avec une graine fixée à 42 et un espace latent de dimension 2.\n\n"
            f"- **Classifieur : {classifier['accuracy']:.2%} de précision**, F1 macro = {classifier['f1_macro']:.4f}.\n"
            f"- **Autoencodeur : MSE de test = {ae['mse']:.5f} par pixel** ; référence image moyenne = {ae['baseline_mse']:.5f}.\n"
            f"- **VAE : MSE de test = {vae['mse_mu']:.5f} par pixel** pour le décodage de la moyenne latente ; "
            f"objectif BCE + KL estimé par {metrics['configuration']['mc_test']} tirages = {vae['loss']:.2f} par image.\n\n"
            "Les tableaux et figures ci-dessous détaillent ces mesures. La MSE des deux autoencodeurs est comparable ; "
            "la perte de classification, la MSE et l'objectif du VAE ont des définitions différentes. "
            "Ces valeurs décrivent l'exécution enregistrée ; une nouvelle exécution recalcule les résultats dans les cellules et fichiers de sortie."
        )
nbformat.validate(notebook)
nbformat.write(notebook, target)
qa = ROOT / ".qa"
qa.mkdir(exist_ok=True)
html, _ = HTMLExporter(template_name="lab").from_notebook_node(notebook)
(qa / "TP2_apercu.html").write_text(html, encoding="utf-8")
print(f"Exécution complète en {time.perf_counter() - started:.1f} s. Notebook et prévisualisation enregistrés.", flush=True)
