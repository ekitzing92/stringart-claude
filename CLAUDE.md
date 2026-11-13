# CLAUDE.md

Consignes de travail sur ce dépôt.

## Le projet

`stringart-claude` calcule une œuvre en *string art* à partir d'une image : une toile ronde, des
clous régulièrement espacés sur le pourtour, un fil noir, et la **séquence de clous** à suivre
pour que la densité des passages restitue l'image.

Deux objectifs, d'égale importance :

1. **L'outil** : le meilleur rendu possible, et une séquence réellement suivable à la main.
2. **Le banc d'essai** : comparer des méthodes d'optimisation sur ce problème, avec une portée
   pédagogique — les échecs comptent autant que les succès et sont consignés dans le carnet.

C'est le pendant « codé par Claude » du dépôt `stringart`, écrit à la main sur le même sujet.
Les deux sont indépendants : pas de code partagé.

## État actuel

La chaîne complète tourne : image → cible → résolution (sept méthodes) → séquence → livrables
(texte, guide HTML, SVG, rendus). Le carnet d'exploration est à jour. Restent à faire, d'après
la feuille de route du README : l'**interface graphique** et l'**affichage en temps réel** de la
résolution (le crochet `suivi` et le journal des solveurs existent déjà pour cela).

```
src/stringart_claude/
├── image.py        chargement, cadrage, mise en tons de la cible
├── toile.py        Toile (physique), Geometrie (clous, cordes, matrice de couverture)
├── moteur.py       Probleme, Etat : modèle, objectif, gains incrémentaux
├── solveurs.py     les méthodes, Resultat, le registre SOLVEURS
├── sequence.py     chemin eulérien, texte de la séquence
├── rendu.py        simulation physique, planches, SVG
├── evaluation.py   le juge (EQM, EQM floue, SSIM)
├── banc.py         le banc d'essai
├── guide.py        guide de montage HTML autonome
├── atelier.py      creer(), Oeuvre : la chaîne complète
└── cli.py          commande `stringart-claude` (creer / banc / algos)
notebooks/exploration.py     le carnet (source) ; exploration.ipynb en est la version exécutée
tests/                       un test_<module>.py par module
```

Mettre cette section, et le README, à jour dans le même mouvement que le code.

## Décisions prises, et pourquoi

Le détail et les mesures sont dans le carnet ; ceci en est l'aide-mémoire.

**Toute image est en valeurs perçues**, tableau 2D de flottants dans [0, 1], 1 = toile blanche.
Cible, prédiction du modèle et simulation se comparent directement.

**Modèle du fil : produit des parts visibles, en valeurs perçues.**
`T[p] = ∏ (1 − c[p,k]) ** (1/GAMMA)`, `GAMMA = 2.2`. Deux alternatives ont été mesurées et
écartées : l'addition écrêtée (biais de −0,18 contre la simulation, elle compte deux fois les
chevauchements) et l'oubli du gamma (le solveur pose moitié moins de fil, la pièce sort
délavée). Le gamma se loge dans la matrice (`c_q = base_q ** (1/gamma)`), il ne coûte rien.

**La simulation (`rendu_physique`) est indépendante du modèle** et sert de juge. Ne jamais la
réécrire à partir du modèle des solveurs, ni l'inverse : c'est leur accord qui valide les deux.
Elle calcule elle-même la surface couverte ; `ImageDraw.line` de Pillow a été essayé et se trompe
jusqu'à 39 % sur la surface d'un trait fin selon son orientation.

**Moteur incrémental.** `Etat` tient à jour le gain de bascule de *toutes* les cordes. Basculer
une corde coûte ~0,3 ms ; tout recalculer, ~30 ms. La matrice est stockée dans les deux sens
(par corde, par pixel), ~140 Mo à 240 clous. Les noyaux sont en numba (`@njit(cache=True)`) ; la
première exécution compile pendant quelques secondes.

**Cordes binaires, sans répétition.** Deux fils sur la même corde se superposent exactement : le
second ne cache rien de plus. Les cordes entre clous trop voisins sont exclues (`Toile.ecart`).

**L'ordre de pose ne change pas l'image.** Les solveurs rendent un *ensemble* de cordes ;
`sequence.sequencer` en fait un fil continu (chemin eulérien, pas de bord pour la parité et la
connexité). Seul `glouton_sequentiel` produit sa séquence lui-même.

**Tous les solveurs partagent moteur, modèle et objectif** : le banc compare des stratégies de
recherche et rien d'autre. Chacun rend un `Resultat` avec son journal (cordes basculées dans
l'ordre), rejouable. Pour en ajouter un : une fonction `nom(probleme, *, cordes_max, suivi,
pas_suivi, ...) -> Resultat`, une fiche dans `SOLVEURS`, un test, une ligne au banc du carnet.

**Méthode recommandée : `recherche_locale`.** D'après le banc, les six méthodes sérieuses tiennent
à moins de 3 % les unes des autres et sont indiscernables à l'œil ; la relaxation continue montre
qu'on est à une dizaine de pour cent d'une borne (non prouvée). Ne pas chercher le gain de rendu
du côté du solveur.

**Ce qui change le rendu**, par ordre d'effet : la mise en tons de la cible (`noir`, `relief`),
le cadrage, le nombre de clous, la finesse du fil. Sans effet mesuré : la résolution de calcul
au-delà de 300 px, l'objectif multi-échelle (`echelles`, gardé dans le code mais inactif).

**Défauts** : 500 mm, 240 clous, fil de 0,2 mm, grille de 300 px, `contraste="local"`,
`relief=0.02`, `noir=0.3`. Le bon `relief` dépend de l'image (0,03 visage, 0,01 photo chargée).

**Numérotation des clous : 0 en haut, sens horaire**, comme un cadran. Dans un tableau NumPy cela
s'écrit `ligne = centre − R·cos θ`, `colonne = centre + R·sin θ`.

## Conventions

- **Français** pour les identifiants (sans accents), les docstrings, les commentaires et les
  messages. Le code doit se lire par quelqu'un qui apprend : dire le *pourquoi*.
- **Tests** : un `tests/test_<module>.py` par module, une fonction de test par fonction testée,
  chaque cas précédé d'un commentaire d'une ligne. Les tests tournent sur une toile miniature
  (`tests/conftest.py`) ; toute la suite prend quelques secondes.
- **Le carnet** : `exploration.py` est la source, `exploration.ipynb` se régénère par
  `uv run jupytext --to ipynb --execute notebooks/exploration.py` (≈ 3 min). Un test vérifie que
  les deux ont le même code. Après toute réexécution, **relire les sorties** : le commentaire du
  carnet cite des chiffres et décrit des images, il doit dire ce qu'elles montrent.
- Pas d'abstraction « au cas où ». Pas de dépendance nouvelle sans besoin constaté.

## Outillage

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format .
uv run --with pyright pyright        # mode standard, zéro signalement
uv run stringart-claude creer images/portrait.png --zoom 1.55 --relief 0.03
```

Un commit part avec les tests au vert, le lint propre et la documentation à jour. Le message dit
pourquoi ; le diff dit déjà quoi.
