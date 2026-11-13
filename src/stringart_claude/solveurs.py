"""Les méthodes de résolution, derrière une interface commune.

Chaque solveur reçoit un `Probleme` et rend un `Resultat` : l'ensemble des cordes retenues, la
séquence de clous qui les parcourt d'un seul fil, et le **journal** de tous les coups joués
(poses et retraits, dans l'ordre). Le journal suffit à rejouer la résolution : c'est lui qui
alimente les courbes de convergence et les instantanés du rendu en cours de route.

Toutes les méthodes partagent le même moteur (`moteur.py`) et donc le même modèle et le même
objectif : ce qu'on compare, c'est bien la stratégie de recherche et rien d'autre.

Le registre `SOLVEURS` décrit chaque méthode et dit lesquelles sont recommandées ; les notes
viennent du banc d'essai (voir `notebooks/exploration.ipynb`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

from .moteur import Etat, Probleme, _basculer
from .sequence import sequencer

# Un coup n'est joué que s'il améliore l'objectif d'au moins cette quantité : sans seuil, le
# bruit d'arrondi pourrait faire tourner une descente indéfiniment.
TOLERANCE = 1e-13

CONVERGENCE = "convergence"  # plus aucun coup n'améliore l'objectif
PLAFOND = "plafond"  # nombre maximal de cordes atteint
IMPASSE = "impasse"  # glouton séquentiel sans saut : plus rien d'utile depuis le clou courant
BUDGET = "budget"  # nombre d'itérations alloué épuisé

Suivi = Callable[[Etat, int], None]


@dataclass
class Resultat:
    """Ce que rend un solveur."""

    nom: str
    cordes: np.ndarray  # numéros des cordes retenues
    sequence: np.ndarray  # clous à visiter dans l'ordre, d'un seul fil
    cout: float  # valeur finale de l'objectif
    duree_s: float
    arret: str
    journal: np.ndarray  # cordes basculées, dans l'ordre (une corde y figure à chaque bascule)
    couts: np.ndarray  # objectif après chaque coup du journal
    cordes_initiales: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    infos: dict = field(default_factory=dict)

    @property
    def n_cordes(self) -> int:
        return int(self.cordes.shape[0])

    def cordes_a(self, n_coups: int) -> np.ndarray:
        """L'ensemble des cordes posées après les `n_coups` premiers coups du journal."""
        return rejouer(self.journal[:n_coups], self.cordes_initiales)


def rejouer(journal: np.ndarray, cordes_initiales=None) -> np.ndarray:
    """Cordes présentes après avoir joué un journal de bascules depuis un ensemble initial."""
    coups = np.asarray(journal, dtype=np.int64)
    if cordes_initiales is not None and len(cordes_initiales):
        coups = np.concatenate([np.asarray(cordes_initiales, dtype=np.int64), coups])
    ids, compte = np.unique(coups, return_counts=True)
    return ids[compte % 2 == 1].astype(np.int32)


class _Journal:
    """Accumule les coups joués, par morceaux."""

    def __init__(self) -> None:
        self._cordes: list[np.ndarray] = []
        self._couts: list[np.ndarray] = []
        self.n = 0

    def ajouter(self, cordes: np.ndarray, couts: np.ndarray, n: int) -> None:
        if n:
            self._cordes.append(cordes[:n].copy())
            self._couts.append(couts[:n].copy())
            self.n += n

    def tableaux(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._cordes:
            return np.empty(0, np.int32), np.empty(0, np.float64)
        return np.concatenate(self._cordes), np.concatenate(self._couts)


def _plafond(cordes_max: int | None, probleme: Probleme) -> int:
    return probleme.geo.n_cordes if cordes_max is None else int(cordes_max)


# --------------------------------------------------------------------------------------------
# Noyaux numba
# --------------------------------------------------------------------------------------------


@njit(cache=True)
def _noyau_meilleur_coup(
    budget, retraits, cordes_max, n_presentes, journal, couts, cout,
    transmission, cible, poids, presente, gain, c_ptr, c_pix, c_q, p_ptr, p_corde, p_q,
):  # fmt: skip
    """Joue jusqu'à `budget` fois le meilleur coup disponible sur toute la toile.

    `retraits=False` : poses seulement (glouton). `retraits=True` : poses et retraits (descente).
    Code de sortie : 0 budget épuisé, 1 convergence, 2 plafond de cordes.
    """
    n_cordes = gain.shape[0]
    fait = 0
    code = 0
    while fait < budget:
        peut_poser = n_presentes < cordes_max
        meilleur = -1
        g_min = -TOLERANCE
        bloque_par_plafond = False
        for k in range(n_cordes):
            g = gain[k]
            if g < g_min:
                if presente[k]:
                    if retraits:
                        g_min = g
                        meilleur = k
                elif peut_poser:
                    g_min = g
                    meilleur = k
                else:
                    bloque_par_plafond = True
        if meilleur < 0:
            code = 2 if bloque_par_plafond else 1
            break
        n_presentes += -1 if presente[meilleur] else 1
        cout += _basculer(
            meilleur, transmission, cible, poids, presente, gain,
            c_ptr, c_pix, c_q, p_ptr, p_corde, p_q,
        )  # fmt: skip
        journal[fait] = meilleur
        couts[fait] = cout
        fait += 1
    return fait, code, cout, n_presentes


@njit(cache=True)
def _noyau_sequentiel(
    budget, clou, cordes_max, n_presentes, id_corde, journal, couts, visites, cout,
    transmission, cible, poids, presente, gain, c_ptr, c_pix, c_q, p_ptr, p_corde, p_q,
):  # fmt: skip
    """Glouton à fil continu : depuis le clou courant, pose la meilleure corde et s'y rend.

    Code de sortie : 0 budget épuisé, 2 plafond de cordes, 3 plus rien d'utile depuis ce clou.
    """
    n_clous = id_corde.shape[0]
    fait = 0
    code = 0
    while fait < budget:
        if n_presentes >= cordes_max:
            code = 2
            break
        meilleur = -1
        suivant = -1
        g_min = -TOLERANCE
        for j in range(n_clous):
            k = id_corde[clou, j]
            if k >= 0 and not presente[k] and gain[k] < g_min:
                g_min = gain[k]
                meilleur = k
                suivant = j
        if meilleur < 0:
            code = 3
            break
        cout += _basculer(
            meilleur, transmission, cible, poids, presente, gain,
            c_ptr, c_pix, c_q, p_ptr, p_corde, p_q,
        )  # fmt: skip
        clou = suivant
        n_presentes += 1
        journal[fait] = meilleur
        couts[fait] = cout
        visites[fait] = clou
        fait += 1
    return fait, code, cout, clou


# Dans du code compilé par numba, le hasard passe par l'ancienne interface `np.random.*` :
# numba lui donne son propre générateur, qu'on initialise ici pour des résultats reproductibles.
@njit(cache=True)
def _semer(graine):
    np.random.seed(graine)  # noqa: NPY002


@njit(cache=True)
def _noyau_recuit(
    i_debut, i_fin, n_total, temp_debut, temp_fin, cordes_max, n_presentes, journal, couts, cout,
    transmission, cible, poids, presente, gain, c_ptr, c_pix, c_q, p_ptr, p_corde, p_q,
):  # fmt: skip
    """Recuit simulé : propose de basculer une corde au hasard, accepte selon Metropolis.

    La température décroît géométriquement de `temp_debut` à `temp_fin` sur `n_total`
    propositions. S'arrête à la proposition `i_fin` ou quand le morceau de journal est plein.
    """
    n_cordes = gain.shape[0]
    capacite = journal.shape[0]
    pente = np.log(temp_fin / temp_debut) / n_total
    fait = 0
    i = i_debut
    while i < i_fin and fait < capacite:
        temperature = temp_debut * np.exp(pente * i)
        i += 1
        k = np.random.randint(n_cordes)  # noqa: NPY002
        if not presente[k] and n_presentes >= cordes_max:
            continue
        g = gain[k]
        if g < 0.0 or np.random.random() < np.exp(-g / temperature):  # noqa: NPY002
            n_presentes += -1 if presente[k] else 1
            cout += _basculer(
                k, transmission, cible, poids, presente, gain,
                c_ptr, c_pix, c_q, p_ptr, p_corde, p_q,
            )  # fmt: skip
            journal[fait] = k
            couts[fait] = cout
            fait += 1
    return i, fait, cout, n_presentes


@njit(cache=True)
def _noyau_tabou(
    budget, iteration, duree_tabou, libre_a, cordes_max, n_presentes, journal, couts, cout,
    meilleur_cout,
    transmission, cible, poids, presente, gain, c_ptr, c_pix, c_q, p_ptr, p_corde, p_q,
):  # fmt: skip
    """Recherche tabou : joue toujours le meilleur coup autorisé, même s'il dégrade l'objectif.

    Une corde basculée est interdite pendant `duree_tabou` itérations (`libre_a[k]` est
    l'itération à partir de laquelle elle redevient jouable), sauf si la jouer bat le record
    (« aspiration »). Rend aussi le rang, dans ce morceau, du dernier record battu (-1 sinon).
    """
    n_cordes = gain.shape[0]
    fait = 0
    rang_record = -1
    while fait < budget:
        peut_poser = n_presentes < cordes_max
        meilleur = -1
        g_min = np.inf
        for k in range(n_cordes):
            g = gain[k]
            if g >= g_min:
                continue
            if not presente[k] and not peut_poser:
                continue
            if libre_a[k] > iteration and cout + g >= meilleur_cout - TOLERANCE:
                continue
            g_min = g
            meilleur = k
        if meilleur < 0:
            break
        n_presentes += -1 if presente[meilleur] else 1
        cout += _basculer(
            meilleur, transmission, cible, poids, presente, gain,
            c_ptr, c_pix, c_q, p_ptr, p_corde, p_q,
        )  # fmt: skip
        libre_a[meilleur] = iteration + duree_tabou + 1
        journal[fait] = meilleur
        couts[fait] = cout
        if cout < meilleur_cout - TOLERANCE:
            meilleur_cout = cout
            rang_record = fait
        fait += 1
        iteration += 1
    return fait, cout, n_presentes, meilleur_cout, rang_record, iteration


@njit(cache=True, parallel=True)
def _relaxation_avant(x, cible, poids, p_ptr, p_corde, p_b):
    """Transmission `exp(−Σ b·x)` de chaque pixel, et dérivée de l'objectif par rapport à Σ."""
    n_pixels = cible.shape[0]
    transmission = np.ones(n_pixels, np.float64)
    derivee = np.zeros(n_pixels, np.float64)
    for p in prange(n_pixels):
        if poids[p] > 0.0:
            s = 0.0
            for y in range(p_ptr[p], p_ptr[p + 1]):
                s += p_b[y] * x[p_corde[y]]
            t = np.exp(-s)
            transmission[p] = t
            derivee[p] = -2.0 * poids[p] * (t - cible[p]) * t
    return transmission, derivee


@njit(cache=True, parallel=True)
def _relaxation_arriere(derivee, c_ptr, c_pix, c_b):
    """Gradient par corde : rétroprojection de la dérivée par pixel."""
    n_cordes = c_ptr.shape[0] - 1
    gradient = np.empty(n_cordes, np.float64)
    for k in prange(n_cordes):
        s = 0.0
        for x in range(c_ptr[k], c_ptr[k + 1]):
            s += c_b[x] * derivee[c_pix[x]]
        gradient[k] = s
    return gradient


# --------------------------------------------------------------------------------------------
# Briques communes
# --------------------------------------------------------------------------------------------


def _jouer_meilleurs_coups(
    etat: Etat,
    journal: _Journal,
    *,
    retraits: bool,
    cordes_max: int,
    suivi: Suivi | None,
    pas_suivi: int,
    coups_max: int | None = None,
) -> str:
    """Enchaîne les meilleurs coups jusqu'à convergence, plafond ou épuisement du budget."""
    n_presentes = etat.n_cordes
    restant = coups_max
    while True:
        budget = pas_suivi if restant is None else min(pas_suivi, restant)
        if budget <= 0:
            return BUDGET
        ids = np.empty(budget, np.int32)
        couts = np.empty(budget, np.float64)
        fait, code, etat.cout, n_presentes = _noyau_meilleur_coup(
            budget, retraits, cordes_max, n_presentes, ids, couts, etat.cout, *etat.arguments()
        )
        journal.ajouter(ids, couts, fait)
        if restant is not None:
            restant -= fait
        if suivi is not None and fait:
            suivi(etat, journal.n)
        if code == 1:
            return CONVERGENCE
        if code == 2:
            return PLAFOND


def _resultat(
    nom: str,
    probleme: Probleme,
    etat: Etat,
    journal: _Journal,
    debut: float,
    arret: str,
    *,
    sequence: np.ndarray | None = None,
    cordes_initiales: np.ndarray | None = None,
    infos: dict | None = None,
) -> Resultat:
    duree = time.perf_counter() - debut
    cordes = np.flatnonzero(etat.presente).astype(np.int32)
    ids, couts = journal.tableaux()
    if sequence is None:
        sequence = sequencer(probleme.geo, cordes)
    # Le coût annoncé est recalculé de zéro : il ne dépend pas de la dérive des mises à jour.
    etat.resynchroniser()
    return Resultat(
        nom=nom,
        cordes=cordes,
        sequence=np.asarray(sequence, dtype=np.int32),
        cout=etat.cout,
        duree_s=duree,
        arret=arret,
        journal=ids,
        couts=couts,
        cordes_initiales=(np.empty(0, np.int32) if cordes_initiales is None else cordes_initiales),
        infos=infos or {},
    )


# --------------------------------------------------------------------------------------------
# Les solveurs
# --------------------------------------------------------------------------------------------


def encre_demandee(probleme: Probleme) -> int:
    """Nombre de cordes qu'il faudrait pour déposer autant d'« encre » que la cible en demande.

    Estimation grossière, par simple bilan de matière : elle ignore que les cordes ne peuvent
    pas déposer l'encre où elles veulent. Sert de budget par défaut au tri naïf.
    """
    geo = probleme.geo
    # −ln(lumière voulue) ; la cible est en valeurs perçues, d'où le facteur gamma.
    profondeur = -geo.gamma * np.log(np.clip(probleme.cible[geo.masque], 0.25, 1.0))
    par_corde = float(-np.log(geo.base_q.astype(np.float64)).sum()) / geo.n_cordes
    return max(1, int(round(float(profondeur.sum()) / par_corde)))


def tri_naif(
    probleme: Probleme,
    *,
    cordes_max: int | None = None,
    suivi: Suivi | None = None,
    pas_suivi: int = 100,
) -> Resultat:
    """Garde les cordes qui traversent les zones les plus sombres, sans se soucier des autres.

    Chaque corde est notée une fois pour toutes par l'obscurité moyenne de la cible le long de
    son trajet ; on pose les mieux notées. C'est la « rétroprojection non filtrée » de la
    tomographie : aucune corde ne sait ce que les autres ont déjà couvert.
    """
    debut = time.perf_counter()
    geo = probleme.geo
    couverture = geo.matrice_couverture()
    obscurite = couverture.T @ (1.0 - probleme.cible.ravel())
    note = obscurite / np.maximum(couverture.T @ np.ones(probleme.cible.size), 1e-12)
    n = min(encre_demandee(probleme), _plafond(cordes_max, probleme))
    choisies = np.argsort(-note, kind="stable")[:n]

    etat = probleme.etat()
    journal = _Journal()
    for i in range(0, n, pas_suivi):
        bloc = choisies[i : i + pas_suivi].astype(np.int32)
        couts = np.empty(bloc.shape[0], np.float64)
        for j, k in enumerate(bloc):
            etat.basculer(int(k))
            couts[j] = etat.cout
        journal.ajouter(bloc, couts, bloc.shape[0])
        if suivi is not None:
            suivi(etat, journal.n)
    return _resultat("tri_naif", probleme, etat, journal, debut, BUDGET)


def glouton_sequentiel(
    probleme: Probleme,
    *,
    cordes_max: int | None = None,
    depart: int = 0,
    saut_bord: bool = True,
    suivi: Suivi | None = None,
    pas_suivi: int = 100,
) -> Resultat:
    """Le glouton historique : le fil repart toujours du dernier clou atteint.

    À chaque pas, on pose la corde issue du clou courant qui réduit le plus l'objectif. Quand
    plus aucune corde utile ne part de ce clou, `saut_bord=True` autorise à longer le bord
    jusqu'au clou le plus proche d'où repart une corde utile ; sinon on s'arrête (« impasse »).
    """
    debut = time.perf_counter()
    geo = probleme.geo
    n_clous = geo.toile.n_clous
    plafond = _plafond(cordes_max, probleme)
    etat = probleme.etat()
    journal = _Journal()
    clou = int(depart) % n_clous
    sequence = [clou]
    existe = geo.id_corde >= 0
    n_sauts = 0
    while True:
        ids = np.empty(pas_suivi, np.int32)
        couts = np.empty(pas_suivi, np.float64)
        visites = np.empty(pas_suivi, np.int32)
        fait, code, etat.cout, clou = _noyau_sequentiel(
            pas_suivi, clou, plafond, journal.n, geo.id_corde, ids, couts, visites, etat.cout,
            *etat.arguments(),
        )  # fmt: skip
        journal.ajouter(ids, couts, fait)
        sequence.extend(visites[:fait].tolist())
        if suivi is not None and fait:
            suivi(etat, journal.n)
        if code == 2:
            arret = PLAFOND
            break
        if code == 3:
            # Meilleur gain disponible au départ de chaque clou.
            gains = np.where(existe & ~etat.presente[geo.id_corde], etat.gain[geo.id_corde], np.inf)
            utiles = np.flatnonzero(gains.min(axis=1) < -TOLERANCE)
            if utiles.shape[0] == 0:
                arret = CONVERGENCE
                break
            if not saut_bord:
                arret = IMPASSE
                break
            # Longer le bord, par le plus court, jusqu'au clou utile le plus proche.
            horaire = (utiles - clou) % n_clous
            trajet = np.minimum(horaire, n_clous - horaire)
            i = int(np.argmin(trajet))
            sens = 1 if horaire[i] <= n_clous - horaire[i] else -1
            for _ in range(int(trajet[i])):
                clou = (clou + sens) % n_clous
                sequence.append(clou)
            n_sauts += 1
    return _resultat(
        "glouton_sequentiel", probleme, etat, journal, debut, arret,
        sequence=np.asarray(sequence, dtype=np.int32), infos={"sauts_de_bord": n_sauts},
    )  # fmt: skip


def glouton_global(
    probleme: Probleme,
    *,
    cordes_max: int | None = None,
    suivi: Suivi | None = None,
    pas_suivi: int = 100,
) -> Resultat:
    """Glouton sans contrainte de continuité : à chaque pas, la meilleure corde de toute la toile.

    Le fil continu est reconstitué après coup par un parcours eulérien (`sequence.py`).
    """
    debut = time.perf_counter()
    etat = probleme.etat()
    journal = _Journal()
    arret = _jouer_meilleurs_coups(
        etat, journal, retraits=False, cordes_max=_plafond(cordes_max, probleme),
        suivi=suivi, pas_suivi=pas_suivi,
    )  # fmt: skip
    return _resultat("glouton_global", probleme, etat, journal, debut, arret)


def recherche_locale(
    probleme: Probleme,
    *,
    cordes_max: int | None = None,
    cordes_initiales=None,
    suivi: Suivi | None = None,
    pas_suivi: int = 100,
) -> Resultat:
    """Glouton avec remords : à chaque pas, le meilleur coup parmi poser *ou retirer* une corde.

    Une corde posée tôt peut devenir nuisible une fois ses voisines en place ; la retirer
    libère de la place pour mieux. On s'arrête sur un optimum local : plus aucune pose ni aucun
    retrait isolé n'améliore l'objectif. Part de zéro, ou de `cordes_initiales`.
    """
    debut = time.perf_counter()
    etat = probleme.etat(cordes_initiales)
    initiales = np.flatnonzero(etat.presente).astype(np.int32)
    journal = _Journal()
    arret = _jouer_meilleurs_coups(
        etat, journal, retraits=True, cordes_max=_plafond(cordes_max, probleme),
        suivi=suivi, pas_suivi=pas_suivi,
    )  # fmt: skip
    return _resultat(
        "recherche_locale", probleme, etat, journal, debut, arret, cordes_initiales=initiales
    )


def recuit(
    probleme: Probleme,
    *,
    cordes_max: int | None = None,
    cordes_initiales=None,
    propositions: int = 3_000_000,
    chaleur: float = 0.3,
    refroidissement: float = 0.01,
    graine: int = 0,
    suivi: Suivi | None = None,
    pas_suivi: int = 100,
) -> Resultat:
    """Recuit simulé sur l'ensemble des cordes, terminé par une descente.

    Part de `cordes_initiales` (par défaut : la solution de la recherche locale) et propose des
    bascules au hasard. Un coup qui dégrade l'objectif de `g` est accepté avec la probabilité
    `exp(−g / température)`. La température initiale vaut `chaleur` fois le gain typique d'un
    coup (médiane des gains de retrait de la solution de départ) et finit `refroidissement`
    fois plus bas. Le résultat rendu est le meilleur entre l'état final et le point de départ.
    """
    debut = time.perf_counter()
    plafond = _plafond(cordes_max, probleme)
    journal = _Journal()
    if cordes_initiales is None:
        etat = probleme.etat()
        _jouer_meilleurs_coups(
            etat, journal, retraits=True, cordes_max=plafond, suivi=suivi, pas_suivi=pas_suivi
        )
    else:
        etat = probleme.etat(cordes_initiales)
    n_avant_recuit = journal.n
    cout_depart = etat.cout
    depart = etat.presente.copy()
    initiales = (
        np.empty(0, np.int32)
        if cordes_initiales is None
        else np.flatnonzero(depart).astype(np.int32)
    )

    posees = etat.gain[etat.presente]
    echelle = (
        float(np.median(np.abs(posees))) if posees.shape[0] else float(np.median(np.abs(etat.gain)))
    )
    temp_debut = max(chaleur * echelle, 1e-300)
    temp_fin = temp_debut * refroidissement

    _semer(graine)
    n_presentes = etat.n_cordes
    i = 0
    while i < propositions:
        ids = np.empty(pas_suivi, np.int32)
        couts = np.empty(pas_suivi, np.float64)
        i, fait, etat.cout, n_presentes = _noyau_recuit(
            i, propositions, propositions, temp_debut, temp_fin, plafond, n_presentes,
            ids, couts, etat.cout, *etat.arguments(),
        )  # fmt: skip
        journal.ajouter(ids, couts, fait)
        if suivi is not None and fait:
            suivi(etat, journal.n)
    # Les gains ont subi des centaines de milliers de corrections : on repart de valeurs exactes.
    etat.resynchroniser()
    arret = _jouer_meilleurs_coups(
        etat, journal, retraits=True, cordes_max=plafond, suivi=suivi, pas_suivi=pas_suivi
    )
    etat.resynchroniser()
    infos = {
        "coups_acceptes": journal.n - n_avant_recuit,
        "cout_depart": cout_depart,
        "temperature_debut": temp_debut,
    }
    if etat.cout > cout_depart:
        # Le recuit n'a pas fait mieux que son point de départ : on rend ce dernier.
        etat = probleme.etat(depart)
        infos["retour_au_depart"] = True
    return _resultat(
        "recuit", probleme, etat, journal, debut, arret, cordes_initiales=initiales, infos=infos
    )


def tabou(
    probleme: Probleme,
    *,
    cordes_max: int | None = None,
    cordes_initiales=None,
    iterations: int = 10_000,
    duree_tabou: int = 100,
    suivi: Suivi | None = None,
    pas_suivi: int = 100,
) -> Resultat:
    """Recherche tabou : une descente qui a le droit de remonter, mais pas de revenir sur ses pas.

    Part de l'optimum local de la recherche locale (ou de celui atteint depuis
    `cordes_initiales`). À chaque itération on joue le meilleur coup autorisé, *même mauvais* ;
    la corde jouée est ensuite interdite pendant `duree_tabou` itérations, ce qui force la
    recherche à s'éloigner au lieu de retomber aussitôt dans le même creux. On rend le meilleur
    état rencontré.
    """
    debut = time.perf_counter()
    plafond = _plafond(cordes_max, probleme)
    etat = probleme.etat(cordes_initiales)
    initiales = np.flatnonzero(etat.presente).astype(np.int32)
    journal = _Journal()
    _jouer_meilleurs_coups(
        etat, journal, retraits=True, cordes_max=plafond, suivi=suivi, pas_suivi=pas_suivi
    )
    cout_depart = etat.cout
    meilleur_cout = etat.cout
    n_record = journal.n  # longueur du journal au moment du meilleur état
    libre_a = np.zeros(probleme.geo.n_cordes, np.int64)
    n_presentes = etat.n_cordes
    iteration = 0
    while iteration < iterations:
        budget = min(pas_suivi, iterations - iteration)
        ids = np.empty(budget, np.int32)
        couts = np.empty(budget, np.float64)
        fait, etat.cout, n_presentes, meilleur_cout, rang, iteration = _noyau_tabou(
            budget, iteration, duree_tabou, libre_a, plafond, n_presentes, ids, couts, etat.cout,
            meilleur_cout, *etat.arguments(),
        )  # fmt: skip
        if rang >= 0:
            n_record = journal.n + rang + 1
        journal.ajouter(ids, couts, fait)
        if suivi is not None and fait:
            suivi(etat, journal.n)
        if fait < budget:
            break
    # On revient au meilleur état rencontré en rejouant le journal jusqu'à lui.
    ids, _ = journal.tableaux()
    etat = probleme.etat(rejouer(ids[:n_record], initiales))
    return _resultat(
        "tabou", probleme, etat, journal, debut, BUDGET, cordes_initiales=initiales,
        infos={"cout_depart": cout_depart, "coups_utiles": n_record},
    )  # fmt: skip


def relaxation(
    probleme: Probleme,
    *,
    cordes_max: int | None = None,
    iterations: int = 300,
    reparer: bool = True,
    suivi: Suivi | None = None,
    pas_suivi: int = 100,
) -> Resultat:
    """Relâche « poser ou non » en « poser un peu », optimise en continu, puis arrondit.

    Chaque corde reçoit une intensité `x ∈ [0, 1]` ; la transmission devient `exp(−Σ b·x)` avec
    `b = −ln q`, qui coïncide avec le modèle quand `x` vaut 0 ou 1. L'objectif est alors lisse
    et se minimise par L-BFGS-B sur les 26 000 intensités à la fois. On arrondit ensuite au
    plus proche (seuil 0,5) et, si `reparer`, une recherche locale répare les dégâts.

    `infos` contient le coût du problème relâché (ce qu'on atteindrait avec des fils
    « partiels ») et celui de l'arrondi brut, pour mesurer ce que l'arrondi fait perdre.
    """
    from scipy.optimize import minimize

    debut = time.perf_counter()
    geo = probleme.geo
    c_b = -np.log(geo.c_q.astype(np.float64)).astype(np.float32)
    p_b = -np.log(geo.p_q.astype(np.float64)).astype(np.float32)
    cible, poids, penalite = probleme.cible_plat, probleme.poids_plat, probleme.penalite_corde

    def objectif(x: np.ndarray) -> tuple[float, np.ndarray]:
        transmission, derivee = _relaxation_avant(x, cible, poids, geo.p_ptr, geo.p_corde, p_b)
        ecart = transmission - cible
        valeur = float(np.sum(poids * ecart * ecart)) + penalite * float(x.sum())
        return valeur, _relaxation_arriere(derivee, geo.c_ptr, geo.c_pix, c_b) + penalite

    solution = minimize(
        objectif,
        np.zeros(geo.n_cordes),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, 1.0)] * geo.n_cordes,
        # Les tolérances par défaut sont absolues : calées sur un objectif d'ordre 1, elles
        # arrêteraient la descente dès les premières itérations sur un coût d'ordre 1e-2.
        options={"maxiter": iterations, "maxfun": 2 * iterations, "ftol": 1e-10, "gtol": 1e-10},
    )
    x = solution.x
    arrondi = np.flatnonzero(x >= 0.5).astype(np.int32)
    plafond = _plafond(cordes_max, probleme)
    if arrondi.shape[0] > plafond:
        arrondi = arrondi[np.argsort(-x[arrondi], kind="stable")[:plafond]]

    etat = probleme.etat(arrondi)
    journal = _Journal()
    infos = {
        "cout_relache": float(solution.fun),
        "cout_arrondi": etat.cout,
        "cordes_arrondi": int(arrondi.shape[0]),
        "cordes_fractionnaires": int(np.count_nonzero((x > 0.02) & (x < 0.98))),
        "iterations": int(solution.nit),
        "intensites": x.astype(np.float32),
    }
    arret = BUDGET
    if reparer:
        arret = _jouer_meilleurs_coups(
            etat, journal, retraits=True, cordes_max=plafond, suivi=suivi, pas_suivi=pas_suivi
        )
    elif suivi is not None:
        suivi(etat, 0)
    return _resultat(
        "relaxation", probleme, etat, journal, debut, arret,
        cordes_initiales=np.sort(arrondi), infos=infos,
    )  # fmt: skip


# --------------------------------------------------------------------------------------------
# Le registre
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Solveur:
    """Fiche d'une méthode : ce qu'elle fait, et ce qu'en dit le banc d'essai."""

    nom: str
    fonction: Callable[..., Resultat]
    resume: str
    qualite: int  # de 0 (à éviter) à 3 (ce qu'on a trouvé de mieux)
    vitesse: int  # de 1 (minutes) à 3 (instantané)
    recommande: bool = False

    @property
    def etoiles(self) -> str:
        return "★" * self.qualite + "☆" * (3 - self.qualite)


# Les notes résument le banc d'essai du notebook (section « Le banc d'essai ») ; les relancer
# avec `stringart-claude banc` si l'on modifie un solveur.
SOLVEURS: dict[str, Solveur] = {
    s.nom: s
    for s in (
        Solveur(
            "recherche_locale", recherche_locale,
            "Glouton avec remords : pose ou retire la meilleure corde jusqu'à l'optimum local.",
            qualite=3, vitesse=3, recommande=True,
        ),
        Solveur(
            "recuit", recuit,
            "Recuit simulé partant de la recherche locale : le dernier pour-cent, en plus lent.",
            qualite=3, vitesse=2,
        ),
        Solveur(
            "tabou", tabou,
            "Recherche tabou partant de la recherche locale : comparable au recuit, déterministe.",
            qualite=3, vitesse=2,
        ),
        Solveur(
            "relaxation", relaxation,
            "Optimisation continue puis arrondi réparé ; fournit aussi une borne de référence.",
            qualite=3, vitesse=1,
        ),
        Solveur(
            "glouton_global", glouton_global,
            "Glouton sans retour en arrière, toutes cordes confondues.",
            qualite=2, vitesse=3,
        ),
        Solveur(
            "glouton_sequentiel", glouton_sequentiel,
            "Glouton historique : le fil repart du dernier clou. Séquence sans parcours eulérien.",
            qualite=2, vitesse=3,
        ),
        Solveur(
            "tri_naif", tri_naif,
            "Témoin négatif : les cordes les plus sombres, sans tenir compte les unes des autres.",
            qualite=0, vitesse=3,
        ),
    )
}  # fmt: skip

RECOMMANDE = "recherche_locale"


def resoudre(probleme: Probleme, solveur: str = RECOMMANDE, **options) -> Resultat:
    """Résout un problème avec la méthode nommée (par défaut : la méthode recommandée)."""
    if solveur not in SOLVEURS:
        raise ValueError(f"solveur inconnu : {solveur!r} (choix : {', '.join(SOLVEURS)})")
    return SOLVEURS[solveur].fonction(probleme, **options)
