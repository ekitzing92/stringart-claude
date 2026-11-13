import json
import re

import pytest

from stringart_claude.guide import guide_html


def _piece(page: str) -> dict:
    """Les données de la pièce embarquées dans la page."""
    trouve = re.search(r"const PIECE = (\{.*?\});", page)
    assert trouve is not None
    return json.loads(trouve.group(1))


def test_guide_html(toile):
    sequence = [0, 20, 21, 40, 5]
    page = guide_html(toile, sequence)

    # La page est autonome et embarque la pièce : séquence, clous, dimensions
    assert page.startswith("<!doctype html>") and "__DONNEES__" not in page
    assert "http://" not in page and "https://" not in page
    donnees = _piece(page)
    assert donnees["sequence"] == sequence
    assert (donnees["clous"], donnees["diametre_mm"], donnees["fil_mm"]) == (48, 200.0, 0.4)

    # Deux pièces différentes n'ont pas la même empreinte (avancement mémorisé séparément)
    autre = _piece(guide_html(toile, sequence[::-1]))
    assert autre["empreinte"] != donnees["empreinte"]

    # Une séquence sans passage n'a pas de guide
    with pytest.raises(ValueError):
        guide_html(toile, [3])
