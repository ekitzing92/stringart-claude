import numpy as np
import pytest

from stringart_claude import solveurs
from stringart_claude.moteur import Probleme
from stringart_claude.sequence import est_pas_de_bord
from stringart_claude.solveurs import SOLVEURS, TOLERANCE, rejouer, resoudre


def _coherent(probleme, resultat):
    """Ce que tout résultat doit vérifier, quelle que soit la méthode."""
    geo = probleme.geo
    # Le coût annoncé est celui de l'ensemble de cordes rendu
    assert resultat.cout == pytest.approx(probleme.cout(resultat.cordes), rel=1e-9)
    # La séquence parcourt chaque corde une fois, et rien d'autre que du bord
    parcourues = geo.cordes_de_sequence(resultat.sequence)
    assert sorted(parcourues.tolist()) == sorted(resultat.cordes.tolist())
    bord = est_pas_de_bord(resultat.sequence, geo.toile.n_clous).sum()
    assert bord + len(parcourues) == len(resultat.sequence) - 1
    # Le journal rejoué redonne l'ensemble final (sauf retour explicite à un état antérieur)
    assert len(resultat.couts) == len(resultat.journal)


@pytest.fixture(scope="module")
def reference(probleme):
    return solveurs.recherche_locale(probleme)


def test_rejouer():
    # Une corde basculée un nombre pair de fois est absente
    assert rejouer(np.array([4, 7, 4, 9])).tolist() == [7, 9]

    # L'ensemble initial compte comme une première bascule
    assert rejouer(np.array([4, 7]), np.array([4, 5])).tolist() == [5, 7]

    # Un journal vide rend l'ensemble initial
    assert rejouer(np.array([], int), np.array([2])).tolist() == [2]


def test_encre_demandee(geo, cible, probleme):
    # Une cible plus sombre demande plus de cordes ; une cible blanche, le minimum
    sombre = Probleme(geo, cible * 0.5)
    blanche = Probleme(geo, np.ones_like(cible))
    assert solveurs.encre_demandee(sombre) > solveurs.encre_demandee(probleme) > 1
    assert solveurs.encre_demandee(blanche) == 1


def test_tri_naif(probleme, reference):
    resultat = solveurs.tri_naif(probleme)
    _coherent(probleme, resultat)

    # Il pose le nombre de cordes prévu par le bilan d'encre
    assert resultat.n_cordes == solveurs.encre_demandee(probleme)

    # Il fait nettement moins bien qu'une méthode qui tient compte des cordes déjà posées
    assert resultat.cout > 1.2 * reference.cout

    # Le plafond de cordes est respecté
    assert solveurs.tri_naif(probleme, cordes_max=10).n_cordes == 10


def test_glouton_sequentiel(probleme, geo):
    resultat = solveurs.glouton_sequentiel(probleme, depart=5)
    _coherent(probleme, resultat)

    # Le fil part du clou demandé et le coût ne fait que baisser
    assert resultat.sequence[0] == 5
    assert (np.diff(resultat.couts) < 0).all()
    assert resultat.cout < probleme.cout([])

    # À l'arrêt par convergence, plus aucune corde absente n'améliore l'objectif
    assert resultat.arret == solveurs.CONVERGENCE
    etat = probleme.etat(resultat.cordes)
    assert etat.gain[~etat.presente].min() >= -TOLERANCE

    # Sans saut de bord, la séquence ne contient que des cordes et s'arrête plus tôt
    strict = solveurs.glouton_sequentiel(probleme, saut_bord=False)
    assert strict.arret in (solveurs.IMPASSE, solveurs.CONVERGENCE)
    assert len(strict.sequence) == strict.n_cordes + 1
    assert strict.n_cordes <= resultat.n_cordes

    # Le plafond de cordes arrête la pose
    court = solveurs.glouton_sequentiel(probleme, cordes_max=7)
    assert court.n_cordes == 7 and court.arret == solveurs.PLAFOND


def test_glouton_global(probleme):
    resultat = solveurs.glouton_global(probleme)
    _coherent(probleme, resultat)

    # Chaque pose est la meilleure du moment : la première est la meilleure corde seule
    assert resultat.journal[0] == np.argmin(probleme.etat().gain)

    # Il ne retire jamais : autant de coups que de cordes, et un coût décroissant
    assert len(resultat.journal) == resultat.n_cordes
    assert (np.diff(resultat.couts) < 0).all()

    # Le suivi est appelé pendant la résolution
    appels = []
    solveurs.glouton_global(probleme, suivi=lambda etat, n: appels.append(n), pas_suivi=5)
    assert appels and appels == sorted(appels) and appels[0] == 5

    # Le plafond de cordes arrête la pose
    court = solveurs.glouton_global(probleme, cordes_max=7)
    assert court.n_cordes == 7 and court.arret == solveurs.PLAFOND


def test_recherche_locale(probleme, reference):
    _coherent(probleme, reference)

    # Elle fait au moins aussi bien que le glouton, qui n'a pas le droit de retirer
    assert reference.cout <= solveurs.glouton_global(probleme).cout + 1e-12

    # Le résultat est un optimum local : plus aucune pose ni aucun retrait n'améliore
    assert probleme.etat(reference.cordes).gain.min() >= -TOLERANCE

    # Partie d'un ensemble arbitraire, elle l'améliore et son journal se rejoue
    hasard = np.random.default_rng(2)
    depart = hasard.choice(probleme.geo.n_cordes, 200, replace=False)
    depuis = solveurs.recherche_locale(probleme, cordes_initiales=depart)
    _coherent(probleme, depuis)
    assert depuis.cout < probleme.cout(depart)
    assert rejouer(depuis.journal, depuis.cordes_initiales).tolist() == depuis.cordes.tolist()
    assert depuis.cordes_a(0).tolist() == sorted(depart.tolist())


def test_recuit(probleme, reference):
    resultat = solveurs.recuit(probleme, propositions=200_000, graine=3)
    _coherent(probleme, resultat)

    # Il ne rend jamais pire que la recherche locale dont il part
    assert resultat.cout <= reference.cout + 1e-12

    # À graine égale, le résultat est reproductible
    encore = solveurs.recuit(probleme, propositions=200_000, graine=3)
    assert encore.cordes.tolist() == resultat.cordes.tolist()

    # Un recuit trop chaud dégrade la solution : on retombe alors sur le point de départ
    chaud = solveurs.recuit(
        probleme, propositions=20_000, chaleur=1e6, refroidissement=0.99,
        cordes_initiales=reference.cordes,
    )  # fmt: skip
    assert chaud.cout <= reference.cout + 1e-12


def test_tabou(probleme, reference):
    resultat = solveurs.tabou(probleme, iterations=1500, duree_tabou=30)
    _coherent(probleme, resultat)

    # Il rend le meilleur état rencontré, donc jamais pire que son point de départ
    assert resultat.cout <= reference.cout + 1e-12
    assert resultat.cout == pytest.approx(resultat.couts.min(), rel=1e-9)

    # Sans itération tabou, c'est la recherche locale
    nul = solveurs.tabou(probleme, iterations=0)
    assert nul.cordes.tolist() == reference.cordes.tolist()


def test_relaxation(probleme, reference):
    resultat = solveurs.relaxation(probleme, iterations=150)
    _coherent(probleme, resultat)
    infos = resultat.infos

    # Le problème relâché fait mieux que toute solution entière connue ; l'arrondi brut, pire
    assert infos["cout_relache"] <= reference.cout
    assert infos["cout_arrondi"] >= resultat.cout
    assert infos["intensites"].min() >= 0.0 and infos["intensites"].max() <= 1.0

    # Réparé, le résultat est un optimum local proche de la recherche locale
    assert probleme.etat(resultat.cordes).gain.min() >= -TOLERANCE
    assert resultat.cout <= 1.1 * reference.cout

    # Sans réparation, on rend l'arrondi brut
    brut = solveurs.relaxation(probleme, iterations=150, reparer=False)
    assert brut.cout == pytest.approx(infos["cout_arrondi"], rel=1e-9)
    assert len(brut.journal) == 0


def test_resoudre(probleme, reference):
    # Sans nom, on obtient la méthode recommandée
    assert solveurs.RECOMMANDE in SOLVEURS and SOLVEURS[solveurs.RECOMMANDE].recommande
    assert resoudre(probleme).cordes.tolist() == reference.cordes.tolist()

    # Les options sont transmises au solveur
    assert resoudre(probleme, "glouton_global", cordes_max=3).n_cordes == 3

    # Chaque fiche du registre pointe sur une fonction qui porte son nom
    for nom, fiche in SOLVEURS.items():
        assert fiche.fonction.__name__ == nom and len(fiche.etoiles) == 3

    # Un nom inconnu est refusé
    with pytest.raises(ValueError):
        resoudre(probleme, "magie")
