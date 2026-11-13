import numpy as np
import pytest

from stringart_claude.moteur import Etat, Probleme, image_modele
from stringart_claude.toile import Geometrie


def test_probleme(toile, geo, cible):
    pb = Probleme(geo, cible)

    # Les poids sont nuls hors du disque et de somme 1 : le coût est une moyenne
    assert pb.poids[~geo.masque].sum() == 0.0
    assert pb.poids.sum() == pytest.approx(1.0)
    attendu = np.mean((1.0 - cible[geo.masque].astype(np.float64)) ** 2)
    assert pb.cout([]) == pytest.approx(attendu)

    # Une carte de poids déplace l'importance sans changer la normalisation
    carte = np.ones_like(cible)
    carte[:32] = 3.0
    pondere = Probleme(geo, cible, poids=carte)
    assert pondere.poids.sum() == pytest.approx(1.0)
    assert pondere.poids[10, 32] == pytest.approx(3 * pondere.poids[50, 32])

    # La pénalité fait payer chaque corde
    cher = Probleme(geo, cible, penalite_corde=0.01)
    assert cher.cout([0, 1, 2]) == pytest.approx(pb.cout([0, 1, 2]) + 0.03)

    # À plusieurs échelles, la cible d'une cellule est la moyenne de ses pixels
    multi = Probleme(Geometrie(toile, echelles=(1, 4)), cible)
    assert multi.cible_plat.shape == (64 * 64 + 16 * 16,)
    assert multi.poids_plat.sum() == pytest.approx(1.0)
    cellule = 64 * 64 + 8 * 16 + 8
    assert multi.cible_plat[cellule] == pytest.approx(cible[32:36, 32:36].mean(), abs=1e-6)

    # Les entrées incohérentes sont refusées
    with pytest.raises(ValueError):
        Probleme(geo, cible[:32])
    with pytest.raises(ValueError):
        Probleme(geo, cible, poids=-carte)
    with pytest.raises(ValueError):
        Probleme(geo, cible, poids_echelles=(1.0, 1.0))


def test_etat(probleme, geo):
    hasard = np.random.default_rng(0)
    etat = probleme.etat()

    # Un état vide est une toile blanche, sans corde
    assert etat.n_cordes == 0
    assert etat.image()[geo.masque] == pytest.approx(1.0)

    # Le gain annoncé pour une corde est exactement la variation de coût constatée
    k = int(np.argmin(etat.gain))
    annonce = etat.gain[k]
    avant = etat.cout
    assert etat.basculer(k) == pytest.approx(annonce)
    assert etat.cout == pytest.approx(avant + annonce)
    assert probleme.cout([k]) == pytest.approx(etat.cout)

    # Après des centaines de bascules, l'état incrémental coïncide avec un recalcul complet
    for corde in hasard.integers(0, geo.n_cordes, 400):
        etat.basculer(int(corde))
    reference = Etat(probleme, etat.presente)
    assert etat.cout == pytest.approx(reference.cout, rel=1e-9)
    assert etat.transmission == pytest.approx(reference.transmission, rel=1e-9)
    assert etat.gain == pytest.approx(reference.gain, rel=1e-6, abs=1e-12)

    # Basculer deux fois la même corde ramène au point de départ
    cout = etat.cout
    etat.basculer(7)
    etat.basculer(7)
    assert etat.cout == pytest.approx(cout)

    # Un état se construit aussi bien d'une liste de cordes que d'un masque
    masque = np.zeros(geo.n_cordes, bool)
    masque[[3, 9]] = True
    assert Etat(probleme, masque).cout == pytest.approx(Etat(probleme, [3, 9]).cout)


def test_image_modele(probleme, geo):
    # Sans corde, la toile est blanche
    assert image_modele(geo, [])[geo.masque] == pytest.approx(1.0)

    # L'image du modèle est celle de l'état du moteur
    cordes = [5, 50, 200, 333]
    assert image_modele(geo, cordes) == pytest.approx(probleme.etat(cordes).image(), abs=1e-6)

    # Une corde n'assombrit que les pixels qu'elle traverse
    image = image_modele(geo, [5])
    touches = np.zeros(geo.toile.cote**2, bool)
    touches[geo.base_pix[geo.base_ptr[5] : geo.base_ptr[6]]] = True
    assert (image.ravel()[touches] < 1.0).all() and (image.ravel()[~touches] == 1.0).all()
