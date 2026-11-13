import numpy as np
import pytest
from PIL import Image

from stringart_claude.cli import construire_analyseur, main
from stringart_claude.solveurs import RECOMMANDE, SOLVEURS

PETITE_TOILE = ["--clous", "48", "--resolution", "64", "--diametre", "200", "--fil", "0.4"]


@pytest.fixture
def photo(tmp_path, cible):
    chemin = tmp_path / "photo.png"
    Image.fromarray((cible * 255).astype(np.uint8)).save(chemin)
    return chemin


def test_construire_analyseur():
    # Les défauts de la commande sont ceux de la bibliothèque
    args = construire_analyseur().parse_args(["creer", "photo.jpg"])
    assert (args.algo, args.clous, args.contraste, args.noir) == (RECOMMANDE, 240, "local", 0.3)

    # Sans sous-commande, la commande refuse de deviner
    with pytest.raises(SystemExit):
        construire_analyseur().parse_args([])


def test_main(photo, tmp_path, capsys):
    # `algos` liste toutes les méthodes et signale celle qu'on recommande
    assert main(["algos"]) == 0
    sortie = capsys.readouterr().out
    assert all(nom in sortie for nom in SOLVEURS) and "recommandée" in sortie

    # `creer` écrit les livrables dans le dossier demandé et résume la pièce
    dossier = tmp_path / "sortie"
    assert main(["creer", str(photo), "-o", str(dossier), "--silencieux", *PETITE_TOILE]) == 0
    assert {f.name for f in dossier.iterdir()} == {
        "rendu.png", "comparaison.png", "sequence.txt", "guide.html", "piece.svg",
    }  # fmt: skip
    assert "cordes" in capsys.readouterr().out

    # `banc` affiche le tableau des mesures puis le classement
    code = main(["banc", str(photo), "--algos", "glouton_global,tri_naif", *PETITE_TOILE])
    sortie = capsys.readouterr().out
    assert code == 0 and "| photo | glouton_global |" in sortie and "retard" in sortie

    # Une image introuvable se solde par un message, pas par une trace d'exécution
    assert main(["creer", str(tmp_path / "absente.png"), "--silencieux"]) == 1
    assert "erreur" in capsys.readouterr().err
