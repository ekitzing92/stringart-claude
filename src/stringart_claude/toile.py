"""La toile : sa géométrie physique, ses clous, ses cordes, et la matrice de couverture.

Conventions, fixées ici une fois pour toutes :

- **Clous** : le clou 0 est en haut, les numéros croissent dans le sens horaire, comme sur un
  cadran. C'est ce qui se lit le plus vite devant la vraie toile.
- **Repère image** : coordonnées continues `(ligne, colonne)`, le pixel `(i, j)` couvre le carré
  `[i, i+1) × [j, j+1)`. Le cercle des clous est inscrit dans le carré : centre `cote / 2`,
  rayon `cote / 2`.
- **Corde** : un segment de fil tendu entre deux clous. Une corde est une paire *non ordonnée*
  `(a, b)` avec `a < b`, identifiée par un entier `k`.
- **Couverture** : `c[p, k]` est la fraction de la surface du pixel `p` cachée par la corde `k`.
  Le fil est bien plus fin qu'un pixel de travail, donc cette fraction vaut
  `longueur traversée × épaisseur du fil / surface du pixel`. C'est le seul endroit où les
  dimensions réelles (mm) entrent dans le calcul.

La matrice de couverture est stockée creuse, deux fois : par corde (quels pixels touche-t-elle ?)
et par pixel (quelles cordes y passent ?). Le moteur incrémental a besoin des deux sens.
On y range `q = 1 − c` (la part du pixel laissée visible) plutôt que `c`, parce que c'est `q`
que le modèle multiplie.

**Perception.** La part visible `∏ q` est une quantité de *lumière*. Pour la comparer à une
image, il faut la ramener en valeur perçue : `(∏ q) ** (1 / GAMMA)`, soit `∏ q ** (1 / GAMMA)`.
Le codage perceptuel se range donc dans la matrice elle-même, sans rien coûter au moteur : les
tableaux `c_q` / `p_q` contiennent `q ** (1 / gamma)`, les tableaux `base_q` le `q` physique.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np
from numba import njit

# L'œil ne perçoit pas la lumière linéairement, et les fichiers image sont codés en conséquence :
# une valeur de pixel `v` désigne une lumière réfléchie de `v ** GAMMA`. La surface cachée par le
# fil, elle, agit sur la lumière. Toute image manipulée par le projet est en valeurs *perçues* ;
# le passage par la physique se fait avec cet exposant.
GAMMA = 2.2

# Une corde ne cache jamais tout un pixel : borne de sécurité pour que q reste > 0
# (le moteur divise par q quand il retire une corde).
COUVERTURE_MAX = 0.95


@dataclass(frozen=True)
class Toile:
    """Description physique de la pièce à réaliser, et résolution du calcul.

    `cote` est la résolution de travail en pixels : c'est l'échelle à laquelle on compare le rendu
    à l'image, pas celle du fil. Avec les valeurs par défaut un pixel mesure ~1,7 mm et un fil
    en couvre 12 % : c'est l'accumulation de fils qui fait le gris.
    """

    diametre_mm: float = 500.0
    n_clous: int = 240
    epaisseur_fil_mm: float = 0.2
    cote: int = 300
    ecart_min: int | None = None

    def __post_init__(self) -> None:
        if self.n_clous < 8:
            raise ValueError("il faut au moins 8 clous")
        if self.cote < 16:
            raise ValueError("résolution de travail trop faible (cote >= 16)")
        if self.diametre_mm <= 0 or self.epaisseur_fil_mm <= 0:
            raise ValueError("les dimensions doivent être strictement positives")
        if self.ecart >= self.n_clous // 2:
            raise ValueError("ecart_min trop grand : aucune corde possible")

    @property
    def ecart(self) -> int:
        """Écart minimal (en clous) entre les deux bouts d'une corde.

        Les cordes entre clous voisins sont minuscules et collées au bord : elles n'apportent
        rien à l'image. Les écarter allège le problème, et réserve le pas de 1 clou au
        « cheminement le long du bord » (voir `sequence.py`).
        """
        if self.ecart_min is not None:
            return max(2, self.ecart_min)
        return max(2, round(self.n_clous / 24))

    @property
    def pas_mm(self) -> float:
        """Taille d'un pixel de travail, en millimètres."""
        return self.diametre_mm / self.cote

    @property
    def alpha(self) -> float:
        """Épaisseur du fil en pixels : part d'un pixel cachée par un fil qui le traverse."""
        return self.epaisseur_fil_mm / self.pas_mm


def positions_clous(n_clous: int, cote: int) -> np.ndarray:
    """Coordonnées `(ligne, colonne)` des clous. Clou 0 en haut, sens horaire."""
    theta = 2.0 * np.pi * np.arange(n_clous) / n_clous
    rayon = cote / 2.0
    centre = cote / 2.0
    # La ligne 0 est en haut de l'image : « horaire à l'écran » s'écrit avec -cos / +sin.
    lignes = centre - rayon * np.cos(theta)
    colonnes = centre + rayon * np.sin(theta)
    return np.stack([lignes, colonnes], axis=1)


def masque_disque(cote: int) -> np.ndarray:
    """Booléens `cote × cote` : vrai pour les pixels dont le centre est dans le cercle des clous."""
    axe = np.arange(cote) + 0.5 - cote / 2.0
    return axe[:, None] ** 2 + axe[None, :] ** 2 <= (cote / 2.0) ** 2


@njit(cache=True)
def _traverser(r0, c0, r1, c1, cote, masque, alpha, pix, q, debut):
    """Parcourt exactement les pixels traversés par le segment (algorithme d'Amanatides-Woo).

    Si `pix` est vide (longueur 0), se contente de compter. Sinon écrit à partir de `debut`
    l'indice à plat de chaque pixel et `q = 1 − couverture`. Rend le nombre de pixels retenus.
    """
    ecrire = pix.shape[0] > 0
    dr = r1 - r0
    dc = c1 - c0
    longueur = np.sqrt(dr * dr + dc * dc)
    if longueur == 0.0:
        return 0

    if dr >= 0:
        i = int(np.floor(r0))
        pas_i = 1
    else:
        i = int(np.ceil(r0)) - 1
        pas_i = -1
    if dc >= 0:
        j = int(np.floor(c0))
        pas_j = 1
    else:
        j = int(np.ceil(c0)) - 1
        pas_j = -1

    if dr > 0:
        t_i = (i + 1 - r0) / dr
        dt_i = 1.0 / dr
    elif dr < 0:
        t_i = (i - r0) / dr
        dt_i = -1.0 / dr
    else:
        t_i = np.inf
        dt_i = np.inf
    if dc > 0:
        t_j = (j + 1 - c0) / dc
        dt_j = 1.0 / dc
    elif dc < 0:
        t_j = (j - c0) / dc
        dt_j = -1.0 / dc
    else:
        t_j = np.inf
        dt_j = np.inf

    n = 0
    t = 0.0
    while t < 1.0:
        t_suiv = min(t_i, t_j, 1.0)
        traverse = (t_suiv - t) * longueur
        if traverse > 1e-9 and 0 <= i < cote and 0 <= j < cote and masque[i, j]:
            if ecrire:
                pix[debut + n] = i * cote + j
                q[debut + n] = 1.0 - min(alpha * traverse, COUVERTURE_MAX)
            n += 1
        if t_suiv >= 1.0:
            break
        if t_i <= t_j:
            i += pas_i
            t_i += dt_i
        else:
            j += pas_j
            t_j += dt_j
        t = t_suiv
    return n


@njit(cache=True)
def _matrice_par_corde(positions, clou_a, clou_b, cote, masque, alpha):
    """Construit la matrice de couverture rangée par corde, en deux passes (compter, remplir)."""
    n_cordes = clou_a.shape[0]
    vide_pix = np.empty(0, np.int32)
    vide_q = np.empty(0, np.float32)
    ptr = np.zeros(n_cordes + 1, np.int64)
    for k in range(n_cordes):
        a, b = clou_a[k], clou_b[k]
        ptr[k + 1] = ptr[k] + _traverser(
            positions[a, 0], positions[a, 1], positions[b, 0], positions[b, 1],
            cote, masque, alpha, vide_pix, vide_q, 0,
        )  # fmt: skip
    pix = np.empty(ptr[n_cordes], np.int32)
    q = np.empty(ptr[n_cordes], np.float32)
    for k in range(n_cordes):
        a, b = clou_a[k], clou_b[k]
        _traverser(
            positions[a, 0], positions[a, 1], positions[b, 0], positions[b, 1],
            cote, masque, alpha, pix, q, ptr[k],
        )  # fmt: skip
    return ptr, pix, q


@njit(cache=True)
def _transposer(ptr, pix, q, n_pixels):
    """Range la même matrice par pixel : pour chaque pixel, les cordes qui y passent."""
    n_cordes = ptr.shape[0] - 1
    p_ptr = np.zeros(n_pixels + 1, np.int64)
    for x in range(pix.shape[0]):
        p_ptr[pix[x] + 1] += 1
    for p in range(n_pixels):
        p_ptr[p + 1] += p_ptr[p]
    curseur = p_ptr[:-1].copy()
    p_corde = np.empty(pix.shape[0], np.int32)
    p_q = np.empty(pix.shape[0], np.float32)
    for k in range(n_cordes):
        for x in range(ptr[k], ptr[k + 1]):
            p = pix[x]
            p_corde[curseur[p]] = k
            p_q[curseur[p]] = q[x]
            curseur[p] += 1
    return p_ptr, p_corde, p_q


@njit(cache=True)
def _agreger(ptr, pix, q, cote, bloc, n_dans_bloc):
    """La même matrice vue à une échelle `bloc` fois plus grossière.

    Les pixels sont regroupés en cellules de `bloc × bloc` ; la couverture d'une cellule est la
    moyenne de celles de ses pixels (ceux du disque). Une corde étant droite, les pixels qu'elle
    traverse dans une même cellule se suivent : il suffit de cumuler les suites consécutives.
    """
    n_cordes = ptr.shape[0] - 1
    cote_g = (cote + bloc - 1) // bloc
    g_ptr = np.zeros(n_cordes + 1, np.int64)
    g_cell = np.empty(pix.shape[0], np.int32)
    g_q = np.empty(pix.shape[0], np.float32)
    n = 0
    for k in range(n_cordes):
        courante = -1
        cumul = 0.0
        for x in range(ptr[k], ptr[k + 1]):
            p = pix[x]
            cellule = (p // cote // bloc) * cote_g + (p % cote) // bloc
            if cellule != courante:
                if courante >= 0:
                    g_cell[n] = courante
                    g_q[n] = 1.0 - min(cumul / n_dans_bloc[courante], COUVERTURE_MAX)
                    n += 1
                courante = cellule
                cumul = 0.0
            cumul += 1.0 - q[x]
        if courante >= 0:
            g_cell[n] = courante
            g_q[n] = 1.0 - min(cumul / n_dans_bloc[courante], COUVERTURE_MAX)
            n += 1
        g_ptr[k + 1] = n
    return g_ptr, g_cell[:n].copy(), g_q[:n].copy()


@njit(cache=True)
def _empiler(ptr_a, pix_a, q_a, ptr_b, pix_b, q_b, decalage_b):
    """Met bout à bout, corde par corde, deux matrices ; les lignes de la seconde sont décalées."""
    n_cordes = ptr_a.shape[0] - 1
    ptr = np.zeros(n_cordes + 1, np.int64)
    pix = np.empty(pix_a.shape[0] + pix_b.shape[0], np.int32)
    q = np.empty(pix.shape[0], np.float32)
    n = 0
    for k in range(n_cordes):
        for x in range(ptr_a[k], ptr_a[k + 1]):
            pix[n] = pix_a[x]
            q[n] = q_a[x]
            n += 1
        for x in range(ptr_b[k], ptr_b[k + 1]):
            pix[n] = pix_b[x] + decalage_b
            q[n] = q_b[x]
            n += 1
        ptr[k + 1] = n
    return ptr, pix, q


class Geometrie:
    """Tout ce qui se déduit d'une `Toile` et ne dépend pas de l'image : clous, cordes, matrice.

    `echelles` liste les niveaux de regroupement des pixels auxquels l'objectif compare rendu et
    cible : `(1,)` compare pixel à pixel ; `(1, 2, 4)` ajoute des cellules de 2×2 et 4×4 pixels,
    où le grain du fil est moyenné comme le fait l'œil à distance. Les cellules de tous les
    niveaux sont numérotées à la suite : pour le moteur, ce ne sont que des « pixels » de plus.
    Les tableaux `base_*` gardent la matrice physique au pixel près, les tableaux `c_*` / `p_*`
    la matrice du moteur : tous niveaux empilés, et `q` élevé à la puissance `1 / gamma`.

    `gamma` vaut `GAMMA` sauf pour l'expérience qui montre ce qu'on perd à l'ignorer (`gamma=1`
    revient à confondre lumière et valeur perçue).

    La construction est coûteuse (de l'ordre de la seconde, une à deux centaines de Mo) : passer
    par `geometrie(toile, ...)`, qui garde en mémoire les dernières toiles utilisées.
    """

    def __init__(
        self, toile: Toile, echelles: tuple[int, ...] = (1,), gamma: float = GAMMA
    ) -> None:
        if not echelles or echelles[0] != 1 or any(b < 1 for b in echelles):
            raise ValueError("echelles doit commencer par 1 et ne contenir que des entiers >= 1")
        if gamma <= 0:
            raise ValueError("gamma doit être strictement positif")
        self.toile = toile
        self.gamma = float(gamma)
        self.echelles = tuple(int(b) for b in echelles)
        n = toile.n_clous
        self.positions = positions_clous(n, toile.cote)
        self.masque = masque_disque(toile.cote)

        a, b = np.triu_indices(n, k=1)
        ecart = np.minimum(b - a, n - (b - a))
        garde = ecart >= toile.ecart
        self.clou_a = a[garde].astype(np.int32)
        self.clou_b = b[garde].astype(np.int32)
        self.n_cordes = int(self.clou_a.shape[0])

        # id_corde[a, b] : numéro de la corde entre a et b, -1 si elle n'existe pas.
        self.id_corde = np.full((n, n), -1, np.int32)
        ids = np.arange(self.n_cordes, dtype=np.int32)
        self.id_corde[self.clou_a, self.clou_b] = ids
        self.id_corde[self.clou_b, self.clou_a] = ids

        cote = toile.cote
        self.base_ptr, self.base_pix, self.base_q = _matrice_par_corde(
            self.positions, self.clou_a, self.clou_b, cote, self.masque, toile.alpha
        )
        # Niveaux de l'objectif. Pour chacun : la taille de bloc, le côté de sa grille de
        # cellules, et le décalage de ses cellules dans la numérotation commune.
        self.niveaux: list[tuple[int, int, int]] = [(1, cote, 0)]
        self.c_ptr, self.c_pix, self.c_q = self.base_ptr, self.base_pix, self.base_q
        self.n_cellules = cote * cote
        for bloc in self.echelles[1:]:
            cote_g = (cote + bloc - 1) // bloc
            g_ptr, g_cell, g_q = _agreger(
                self.base_ptr, self.base_pix, self.base_q, cote, bloc,
                self.regrouper(self.masque.astype(np.float64), bloc).ravel(),
            )  # fmt: skip
            self.c_ptr, self.c_pix, self.c_q = _empiler(
                self.c_ptr, self.c_pix, self.c_q, g_ptr, g_cell, g_q, self.n_cellules
            )
            self.niveaux.append((bloc, cote_g, self.n_cellules))
            self.n_cellules += cote_g * cote_g
        if self.gamma != 1.0:
            self.c_q = (self.c_q.astype(np.float64) ** (1.0 / self.gamma)).astype(np.float32)
        self.p_ptr, self.p_corde, self.p_q = _transposer(
            self.c_ptr, self.c_pix, self.c_q, self.n_cellules
        )

    def regrouper(self, champ: np.ndarray, bloc: int) -> np.ndarray:
        """Somme d'un champ `cote × cote` sur des blocs de `bloc × bloc` pixels."""
        cote = self.toile.cote
        cote_g = (cote + bloc - 1) // bloc
        complet = np.zeros((cote_g * bloc, cote_g * bloc), np.float64)
        complet[:cote, :cote] = champ
        return complet.reshape(cote_g, bloc, cote_g, bloc).sum(axis=(1, 3))

    @cached_property
    def longueur_mm(self) -> np.ndarray:
        """Longueur réelle de chaque corde, en millimètres."""
        delta = self.positions[self.clou_a] - self.positions[self.clou_b]
        return np.hypot(delta[:, 0], delta[:, 1]) * self.toile.pas_mm

    @property
    def n_coefficients(self) -> int:
        return int(self.c_pix.shape[0])

    def matrice_couverture(self):
        """La matrice physique `c[p, k]` au format `scipy.sparse.csc_matrix` (pixels × cordes).

        Pour l'analyse et le notebook ; les solveurs travaillent sur les tableaux bruts.
        """
        from scipy.sparse import csc_matrix

        n_pixels = self.toile.cote * self.toile.cote
        return csc_matrix(
            (1.0 - self.base_q, self.base_pix, self.base_ptr), shape=(n_pixels, self.n_cordes)
        )

    def cordes_de_sequence(self, sequence) -> np.ndarray:
        """Numéros des cordes parcourues par une séquence de clous (les pas de bord sont omis)."""
        seq = np.asarray(sequence, dtype=np.int64)
        if seq.shape[0] < 2:
            return np.empty(0, np.int32)
        ids = self.id_corde[seq[:-1], seq[1:]]
        return ids[ids >= 0]


@lru_cache(maxsize=3)
def geometrie(toile: Toile, echelles: tuple[int, ...] = (1,), gamma: float = GAMMA) -> Geometrie:
    """La géométrie d'une toile, mise en cache (les toiles identiques la partagent)."""
    return Geometrie(toile, echelles, gamma)
