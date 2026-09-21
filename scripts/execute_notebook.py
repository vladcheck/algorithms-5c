"""Выполнение ноутбука с помощью nbclient (headless)."""

import os
import sys

import nbformat as nbf
from nbclient import NotebookClient

path = sys.argv[1] if len(sys.argv) > 1 else "notebooks/Листок-1.ipynb"
nb = nbf.read(path, as_version=4)
resources = {"metadata": {"path": os.path.dirname(os.path.abspath(path))}}
client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources=resources)
client.execute()
nbf.write(nb, path)
print(path, "executed")
