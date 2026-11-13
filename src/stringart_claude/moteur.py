"""Le modèle du fil, la fonction objectif, et leur évaluation incrémentale.

**Modèle.** Un fil est opaque et bien plus fin qu'un pixel de travail. S'il cache une fraction
`c` du pixel, il en laisse voir `q = 1 − c`. Deux fils qui se croisent se recouvrent en partie :
la part de toile encore visible est le *produit* des parts laissées par chacun,

    lumière[p] = ∏ q[p, k]   pour les cordes k posées      (1 = toile blanche, 0 = noir)

C'est une loi de Beer-Lambert discrète. Elle sature naturellement — on tend vers le noir sans
jamais le dépasser. Ce produit est une quantité de lumière ; ce que l'œil en perçoit, et ce à
quoi une image se compare, est sa racine `GAMMA`-ième (voir `toile.py`). Le moteur travaille
directement sur cette valeur perçue, qu'on appelle ici **transmission** :

    T[p] = ∏ q[p, k] ** (1 / gamma)

**Objectif.** L'écart quadratique pondéré entre la transmission et l'image cible :

    coût(S) = Σ_p W[p] · (T[p] − L[p])²  +  λ · |S|

`W` est normalisé à somme 1 sur le disque : sans pénalité, le coût est une erreur quadratique
moyenne en valeurs perçues. `λ` (optionnel) fait payer chaque corde, pour troquer un peu de
fidélité contre des heures de montage.

**Gains.** Tout l'intérêt du moteur : on tient à jour, pour *chaque* corde, la variation de coût
que provoquerait son basculement (la poser si elle est absente, la retirer sinon). Basculer une
corde ne modifie que les pixels qu'elle traverse ; seuls les gains des cordes passant par ces
pixels sont à corriger. Choisir le meilleur coup parmi 26 000 devient un simple `argmin`, et un
basculement coûte quelques dixièmes de milliseconde au lieu d'une réévaluation complète.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .toile import Geometrie


@njit(cache=True)
def _variation(t, q, cible, presente):
    """Variation de (T − L)² sur un pixel si l'on bascule une corde de part visible `q`."""
    t2 = t / q if presente else t * q
    return (t2 - t) * (t2 + t - 2.0 * cible)


@njit(cache=True)
def _gains_complets(transmission, cible, poids, presente, c_ptr, c_pix, c_q, penalite):
    """Recalcule tous les gains de basculement à partir de zéro."""
    n_cordes = c_ptr.shape[0] - 1
    gain = np.empty(n_cordes, np.float64)
    for k in range(n_cordes):
        somme = 0.0
        pres = presente[k]
        for x in range(c_ptr[k], c_ptr[k + 1]):
            p = c_pix[x]
            somme += poids[p] * _variation(transmission[p], c_q[x], cible[p], pres)
        gain[k] = somme - penalite if pres else somme + penalite
    return gain


@njit(cache=True)
def _basculer(
    k, transmission, cible, poids, presente, gain, c_ptr, c_pix, c_q, p_ptr, p_corde, p_q
):
    """Pose ou retire la corde `k` et corrige les gains de toutes les cordes qu'elle croise.

    Rend la variation de l'objectif (pénalité comprise).
    """
    ancien = gain[k]
    ajout = not presente[k]
    for x in range(c_ptr[k], c_ptr[k + 1]):
        p = c_pix[x]
        t_avant = transmission[p]
        t_apres = t_avant * c_q[x] if ajout else t_avant / c_q[x]
        w = poids[p]
        if w > 0.0:
            deux_l = 2.0 * cible[p]
            for y in range(p_ptr[p], p_ptr[p + 1]):
                m = p_corde[y]
                qm = p_q[y]
                if presente[m]:
                    a_avant = t_avant / qm
                    a_apres = t_apres / qm
                else:
                    a_avant = t_avant * qm
                    a_apres = t_apres * qm
                gain[m] += w * (
                    (a_apres - t_apres) * (a_apres + t_apres - deux_l)
                    - (a_avant - t_avant) * (a_avant + t_avant - deux_l)
                )
        transmission[p] = t_apres
    presente[k] = ajout
    # Rebasculer k annulerait exactement ce qu'on vient de faire.
    gain[k] = -ancien
    return ancien


@njit(cache=True)
def _transmission_de(presente, c_ptr, c_pix, c_q, n_pixels):
    """Transmission produite par un ensemble de cordes, recalculée de zéro."""
    transmission = np.ones(n_pixels, np.float64)
    for k in range(presente.shape[0]):
        if presente[k]:
            for x in range(c_ptr[k], c_ptr[k + 1]):
                transmission[c_pix[x]] *= c_q[x]
    return transmission


def masque_cordes(n_cordes: int, cordes) -> np.ndarray:
    """Un ensemble de cordes (numéros, masque booléen ou rien) sous forme de masque booléen."""
    presente = np.zeros(n_cordes, np.bool_)
    if cordes is None:
        return presente
    cordes = np.asarray(cordes)
    if cordes.dtype == np.bool_:
        if cordes.shape != (n_cordes,):
            raise ValueError(f"masque de {cordes.shape[0]} cordes, attendu {n_cordes}")
        presente[:] = cordes
    elif cordes.size:
        presente[cordes.astype(np.int64)] = True
    return presente


class Probleme:
    """Une toile, une image cible, des poids : tout ce qu'un solveur doit connaître.

    `cible` est une luminance `cote × cote` dans [0, 1] (1 = blanc). `poids` est une carte
    d'importance optionnelle, de même forme ; elle est annulée hors du disque.

    Si la géométrie porte plusieurs échelles, l'objectif est la moyenne pondérée des écarts
    quadratiques moyens mesurés à chacune : `poids_echelles` donne la part de chaque niveau
    (par défaut, parts égales). La cible d'une cellule est la moyenne de celle de ses pixels.
    """

    def __init__(
        self,
        geo: Geometrie,
        cible: np.ndarray,
        poids: np.ndarray | None = None,
        penalite_corde: float = 0.0,
        poids_echelles: tuple[float, ...] | None = None,
    ) -> None:
        cote = geo.toile.cote
        cible = np.clip(np.asarray(cible, dtype=np.float64), 0.0, 1.0)
        if cible.shape != (cote, cote):
            raise ValueError(f"cible de forme {cible.shape}, attendu {(cote, cote)}")
        carte = np.ones((cote, cote)) if poids is None else np.asarray(poids, dtype=np.float64)
        if carte.shape != (cote, cote) or carte.min() < 0:
            raise ValueError("poids invalides : même forme que la cible, et positifs")
        carte = np.where(geo.masque, carte, 0.0)
        if carte.sum() <= 0:
            raise ValueError("poids tous nuls dans le disque")
        carte = carte / carte.sum()
        if poids_echelles is None:
            poids_echelles = (1.0,) * len(geo.niveaux)
        if len(poids_echelles) != len(geo.niveaux) or min(poids_echelles) < 0:
            raise ValueError("poids_echelles : un poids positif par échelle de la géométrie")
        parts = np.asarray(poids_echelles, dtype=np.float64)
        parts = parts / parts.sum()

        self.geo = geo
        self.toile = geo.toile
        self.cible = cible
        self.poids: np.ndarray = carte
        self.penalite_corde = float(penalite_corde)
        # Vues à plat, tous niveaux à la suite : celles que manipulent les noyaux numba.
        cibles, poids_niveaux = [], []
        for (bloc, _, _), part in zip(geo.niveaux, parts, strict=True):
            masse = geo.regrouper(carte, bloc)
            moyenne = geo.regrouper(carte * cible, bloc) / np.maximum(masse, 1e-300)
            cibles.append(np.where(masse > 0, moyenne, 1.0).ravel())
            poids_niveaux.append(part * masse.ravel())
        self.cible_plat = np.ascontiguousarray(np.concatenate(cibles))
        self.poids_plat = np.ascontiguousarray(np.concatenate(poids_niveaux))

    def etat(self, cordes=None) -> Etat:
        """Un état de recherche neuf, vide ou initialisé avec un ensemble de cordes."""
        return Etat(self, cordes)

    def cout(self, cordes) -> float:
        """Valeur de l'objectif pour un ensemble de cordes."""
        return self.etat(cordes).cout


class Etat:
    """L'état courant d'une recherche : cordes posées, transmission, gains de basculement.

    Les solveurs écrits en numba reçoivent directement les tableaux (`arguments()`), puis
    appellent `resynchroniser()` ; le reste du code passe par `basculer`.
    """

    def __init__(self, probleme: Probleme, cordes=None) -> None:
        geo = probleme.geo
        self.probleme = probleme
        self.presente = masque_cordes(geo.n_cordes, cordes)
        self.transmission = np.empty(0)
        self.gain = np.empty(0)
        self.cout = 0.0
        self.resynchroniser()

    def resynchroniser(self) -> None:
        """Recalcule transmission, gains et coût à partir du seul ensemble de cordes.

        Sert à l'initialisation, après un solveur numba, et de garde-fou contre la dérive
        numérique des mises à jour incrémentales.
        """
        pb, geo = self.probleme, self.probleme.geo
        self.transmission = _transmission_de(
            self.presente, geo.c_ptr, geo.c_pix, geo.c_q, geo.n_cellules
        )
        self.gain = _gains_complets(
            self.transmission, pb.cible_plat, pb.poids_plat, self.presente,
            geo.c_ptr, geo.c_pix, geo.c_q, pb.penalite_corde,
        )  # fmt: skip
        ecart = self.transmission - pb.cible_plat
        self.cout = float(np.sum(pb.poids_plat * ecart * ecart, dtype=np.float64)) + (
            pb.penalite_corde * self.n_cordes
        )

    @property
    def n_cordes(self) -> int:
        return int(self.presente.sum())

    def basculer(self, k: int) -> float:
        """Pose la corde `k` si elle est absente, la retire sinon. Rend la variation du coût."""
        pb, geo = self.probleme, self.probleme.geo
        variation = _basculer(
            k, self.transmission, pb.cible_plat, pb.poids_plat, self.presente, self.gain,
            geo.c_ptr, geo.c_pix, geo.c_q, geo.p_ptr, geo.p_corde, geo.p_q,
        )  # fmt: skip
        self.cout += variation
        return variation

    def arguments(self) -> tuple:
        """Les tableaux attendus, dans l'ordre, par les noyaux numba des solveurs."""
        pb, geo = self.probleme, self.probleme.geo
        return (
            self.transmission, pb.cible_plat, pb.poids_plat, self.presente, self.gain,
            geo.c_ptr, geo.c_pix, geo.c_q, geo.p_ptr, geo.p_corde, geo.p_q,
        )  # fmt: skip

    def image(self) -> np.ndarray:
        """La luminance prédite par le modèle, `cote × cote` (niveau le plus fin)."""
        cote = self.probleme.toile.cote
        return self.transmission[: cote * cote].reshape(cote, cote).astype(np.float32)


def image_modele(geo: Geometrie, cordes) -> np.ndarray:
    """Image perçue que prédit le modèle pour un ensemble de cordes."""
    presente = masque_cordes(geo.n_cordes, cordes)
    cote = geo.toile.cote
    lumiere = _transmission_de(presente, geo.base_ptr, geo.base_pix, geo.base_q, cote * cote)
    return (lumiere ** (1.0 / geo.gamma)).reshape(cote, cote).astype(np.float32)
