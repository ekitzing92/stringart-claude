"""Garde-fous du carnet `notebooks/exploration.py` : il ne doit pas se périmer en silence."""

import ast
import json
from pathlib import Path

import jupytext

import stringart_claude

CARNET = Path(__file__).parent.parent / "notebooks" / "exploration.py"


def test_exploration():
    arbre = ast.parse(CARNET.read_text("utf-8"))

    # Tout ce que le carnet importe du paquet existe encore
    importes: list[tuple[str, str]] = []
    for noeud in ast.walk(arbre):
        if (
            isinstance(noeud, ast.ImportFrom)
            and noeud.module
            and noeud.module.startswith("stringart_claude")
        ):
            importes += [(noeud.module, alias.name) for alias in noeud.names]
    assert len(importes) > 10
    for module, nom in importes:
        source = stringart_claude
        for partie in module.split(".")[1:]:
            source = getattr(source, partie)
        assert hasattr(source, nom), f"{module}.{nom} n'existe plus"

    # Les méthodes que le carnet appelle par leur nom sont toujours au registre
    chaines = {n.value for n in ast.walk(arbre) if isinstance(n, ast.Constant)}
    for nom in ("tri_naif", "glouton_sequentiel", "glouton_global", "recherche_locale", "tabou"):
        assert nom in chaines and nom in stringart_claude.SOLVEURS

    # La version exécutée (.ipynb) contient exactement le code du .py : on la régénère après
    # chaque modification, sans quoi ses sorties ne correspondraient plus à ce qu'on lit
    execute = json.loads(CARNET.with_suffix(".ipynb").read_text("utf-8"))
    code_execute = ["".join(c["source"]) for c in execute["cells"] if c["cell_type"] == "code"]
    code_source = [c.source for c in jupytext.read(CARNET).cells if c.cell_type == "code"]
    assert code_execute == code_source
