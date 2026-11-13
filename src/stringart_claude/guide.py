"""Le guide de montage : la séquence, pas à pas, dans une page HTML autonome.

Devant la toile on a les deux mains prises et les yeux sur les clous. La page affiche donc un
seul gros numéro — le prochain clou —, peut le dire à voix haute, avancer toute seule, et retient
où l'on s'est arrêté. Elle n'a besoin ni de réseau ni de serveur : on ouvre le fichier.
"""

from __future__ import annotations

import json

from .sequence import longueur_fil_m
from .toile import Toile

_GABARIT = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Guide de montage — string art</title>
<style>
  :root {
    --fond: #f4f1ea; --carte: #fffdf8; --encre: #1d1b18; --doux: #6f685d; --trait: #d9d2c4;
    --accent: #b3401f; --accent-encre: #fff;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --fond: #171614; --carte: #211f1c; --encre: #efe9dd; --doux: #a39a8a; --trait: #3a362f;
      --accent: #e8703f; --accent-encre: #1b1a17;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--fond); color: var(--encre);
    font: 16px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  main {
    max-width: 1080px; margin: 0 auto; padding: 16px;
    display: grid; gap: 16px; grid-template-columns: minmax(0, 1fr);
  }
  @media (min-width: 820px) {
    main { grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); align-items: start; }
  }
  header { grid-column: 1 / -1; }
  h1 { font-size: 1.15rem; margin: 0 0 2px; letter-spacing: .02em; }
  .infos { color: var(--doux); font-size: .88rem; }
  .carte { background: var(--carte); border: 1px solid var(--trait); border-radius: 14px; }
  .toile { padding: 10px; }
  canvas { display: block; width: 100%; height: auto; aspect-ratio: 1; }
  .pas { padding: 18px; text-align: center; }
  .rang { color: var(--doux); font-variant-numeric: tabular-nums; }
  .trajet { color: var(--doux); margin-top: 10px; font-size: 1.05rem; }
  .clou {
    font-size: clamp(5rem, 26vw, 9.5rem); font-weight: 700; line-height: 1;
    font-variant-numeric: tabular-nums; margin: 4px 0;
  }
  .bord {
    display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .85rem;
    background: var(--accent); color: var(--accent-encre); visibility: hidden;
  }
  .bord.visible { visibility: visible; }
  .suite { color: var(--doux); font-variant-numeric: tabular-nums; min-height: 1.4em; }
  progress { width: 100%; height: 8px; margin-top: 14px; accent-color: var(--accent); }
  .boutons { display: grid; grid-template-columns: 1fr 2fr; gap: 10px; margin-top: 14px; }
  button, select, input {
    font: inherit; color: var(--encre); background: var(--fond);
    border: 1px solid var(--trait); border-radius: 10px; padding: 10px 12px;
  }
  button { cursor: pointer; }
  button:active { transform: translateY(1px); }
  #suivant {
    background: var(--accent); color: var(--accent-encre); border-color: var(--accent);
    font-weight: 600; font-size: 1.15rem; padding: 16px;
  }
  #precedent { font-size: 1.05rem; }
  .reglages {
    display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; justify-content: center;
    margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--trait); font-size: .92rem;
  }
  .reglages label { display: inline-flex; align-items: center; gap: 6px; }
  .reglages input[type=number] { width: 6.5em; padding: 6px 8px; }
  .reglages select { padding: 6px 8px; }
  .aide { grid-column: 1 / -1; color: var(--doux); font-size: .85rem; }
  kbd {
    border: 1px solid var(--trait); border-radius: 5px; padding: 0 5px; background: var(--carte);
  }
</style>
</head>
<body>
<main>
  <header>
    <h1>Guide de montage</h1>
    <div class="infos" id="infos"></div>
  </header>

  <section class="carte toile"><canvas id="toile" width="900" height="900"></canvas></section>

  <section class="carte pas">
    <div class="rang" id="rang"></div>
    <div class="trajet" id="trajet"></div>
    <div class="clou" id="clou" aria-live="polite"></div>
    <div><span class="bord" id="bord">longer le bord</span></div>
    <div class="suite" id="suite"></div>
    <progress id="avancement" value="0" max="1"></progress>
    <div class="boutons">
      <button id="precedent">◀ Retour</button>
      <button id="suivant">Fait, suivant ▶</button>
    </div>
    <div class="reglages">
      <label><input type="checkbox" id="voix"> Dire le numéro</label>
      <label><input type="checkbox" id="auto"> Avancer seul toutes les
        <select id="delai">
          <option value="3">3 s</option><option value="5">5 s</option>
          <option value="8" selected>8 s</option><option value="12">12 s</option>
          <option value="20">20 s</option>
        </select>
      </label>
      <label>Aller au passage <input type="number" id="aller" min="1"></label>
    </div>
  </section>

  <p class="aide">
    Le clou 0 est en haut ; les numéros croissent dans le sens des aiguilles d'une montre.
    Le grand nombre est le clou à rejoindre. Clavier : <kbd>→</kbd> ou <kbd>Espace</kbd> pour
    avancer, <kbd>←</kbd> pour revenir. L'avancement est mémorisé sur cet appareil.
  </p>
</main>

<script>
const PIECE = __DONNEES__;
const SEQ = PIECE.sequence, N = PIECE.clous, TOTAL = SEQ.length - 1;
const $ = (id) => document.getElementById(id);
const CLE = "stringart-guide-" + PIECE.empreinte;

// i = nombre de passages déjà faits ; le passage en cours va de SEQ[i] à SEQ[i + 1].
let i = 0;
try { i = Math.min(Math.max(parseInt(localStorage.getItem(CLE) || "0", 10) || 0, 0), TOTAL); }
catch (e) { /* stockage indisponible : on repart du début */ }

const toile = $("toile"), ctx = toile.getContext("2d");
const fond = document.createElement("canvas");   // les fils déjà posés, dessinés une fois
const fctx = fond.getContext("2d");
let cote = 900, poses = -1;

function position(clou, rayon) {
  const a = 2 * Math.PI * clou / N;
  return [cote / 2 + rayon * Math.sin(a), cote / 2 - rayon * Math.cos(a)];
}
const estBord = (a, b) => { const d = Math.abs(a - b) % N; return Math.min(d, N - d) === 1; };

function tailler() {
  const dpr = window.devicePixelRatio || 1;
  cote = Math.round(toile.clientWidth * dpr) || 900;
  toile.width = toile.height = fond.width = fond.height = cote;
  poses = -1;
  dessiner();
}

function tracerFond(jusqua) {
  const R = cote * 0.44;
  if (poses < 0 || jusqua < poses) {
    fctx.fillStyle = "#fff";
    fctx.clearRect(0, 0, cote, cote);
    fctx.beginPath(); fctx.arc(cote / 2, cote / 2, R, 0, 2 * Math.PI); fctx.fill();
    poses = 0;
  }
  fctx.strokeStyle = "rgba(0,0,0,0.55)";
  fctx.lineWidth = Math.max(0.35, PIECE.fil_mm / PIECE.diametre_mm * 2 * R);
  fctx.beginPath();
  for (let k = poses; k < jusqua; k++) {
    const [x0, y0] = position(SEQ[k], R), [x1, y1] = position(SEQ[k + 1], R);
    fctx.moveTo(x0, y0); fctx.lineTo(x1, y1);
  }
  fctx.stroke();
  poses = jusqua;
}

function dessiner() {
  const R = cote * 0.44, style = getComputedStyle(document.documentElement);
  const doux = style.getPropertyValue("--doux"), accent = style.getPropertyValue("--accent");
  tracerFond(i);
  ctx.clearRect(0, 0, cote, cote);
  ctx.drawImage(fond, 0, 0);
  // Les clous et leurs repères.
  const pasRepere = N <= 120 ? 5 : 10;
  ctx.fillStyle = doux; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.font = Math.round(cote * 0.022) + "px system-ui, sans-serif";
  for (let c = 0; c < N; c++) {
    const [x, y] = position(c, R);
    ctx.beginPath(); ctx.arc(x, y, cote * (c % pasRepere ? 0.002 : 0.0035), 0, 2 * Math.PI);
    ctx.fill();
    if (c % pasRepere === 0) {
      const [tx, ty] = position(c, R + cote * 0.033); ctx.fillText(c, tx, ty);
    }
  }
  if (i < TOTAL) {
    const [x0, y0] = position(SEQ[i], R), [x1, y1] = position(SEQ[i + 1], R);
    ctx.strokeStyle = accent; ctx.lineWidth = cote * 0.004; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.fillStyle = accent;
    ctx.beginPath(); ctx.arc(x1, y1, cote * 0.011, 0, 2 * Math.PI); ctx.fill();
    ctx.beginPath(); ctx.arc(x0, y0, cote * 0.006, 0, 2 * Math.PI); ctx.fill();
  }
}

function afficher(parler) {
  const fini = i >= TOTAL;
  $("rang").textContent = fini ? "Terminé" : "Passage " + (i + 1) + " sur " + TOTAL;
  $("trajet").textContent = fini
    ? "Nouer le fil au clou " + SEQ[TOTAL] : "du clou " + SEQ[i] + " au clou";
  $("clou").textContent = fini ? "✓" : SEQ[i + 1];
  $("bord").classList.toggle("visible", !fini && estBord(SEQ[i], SEQ[i + 1]));
  $("suite").textContent = fini
    ? "" : "Ensuite : " + (SEQ.slice(i + 2, i + 7).join(" · ") || "fin");
  $("avancement").max = TOTAL; $("avancement").value = i;
  $("aller").max = TOTAL; $("aller").value = Math.min(i + 1, TOTAL);
  try { localStorage.setItem(CLE, String(i)); } catch (e) { /* tant pis */ }
  dessiner();
  if (parler && !fini && $("voix").checked && "speechSynthesis" in window) {
    speechSynthesis.cancel();
    const dit = new SpeechSynthesisUtterance(
      (estBord(SEQ[i], SEQ[i + 1]) ? "bord, " : "") + SEQ[i + 1]);
    dit.lang = "fr-FR"; dit.rate = 0.95;
    speechSynthesis.speak(dit);
  }
}

function aller(k, parler = true) { i = Math.min(Math.max(k, 0), TOTAL); afficher(parler); }

let minuterie = null, verrou = null;
async function reglerAuto() {
  clearInterval(minuterie); minuterie = null;
  if ($("auto").checked) {
    minuterie = setInterval(() => {
      if (i >= TOTAL) { $("auto").checked = false; reglerAuto(); } else aller(i + 1);
    }, 1000 * parseInt($("delai").value, 10));
    try { if ("wakeLock" in navigator) verrou = await navigator.wakeLock.request("screen"); }
    catch (e) { /* l'écran pourra se mettre en veille */ }
  } else if (verrou) { try { await verrou.release(); } catch (e) {} verrou = null; }
}

$("suivant").addEventListener("click", () => aller(i + 1));
$("precedent").addEventListener("click", () => aller(i - 1));
$("auto").addEventListener("change", reglerAuto);
$("delai").addEventListener("change", reglerAuto);
$("voix").addEventListener("change", () => afficher(true));
$("aller").addEventListener("change", (e) => aller((parseInt(e.target.value, 10) || 1) - 1));
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "ArrowRight" || e.key === " ") { e.preventDefault(); aller(i + 1); }
  if (e.key === "ArrowLeft") { e.preventDefault(); aller(i - 1); }
});
window.addEventListener("resize", tailler);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", dessiner);

$("infos").textContent = "Toile de " + PIECE.diametre_mm + " mm · " + N + " clous · fil de "
  + PIECE.fil_mm + " mm · " + TOTAL + " passages · environ " + PIECE.fil_m + " m de fil";
tailler();
afficher(false);
</script>
</body>
</html>
"""


def guide_html(toile: Toile, sequence) -> str:
    """Rend la page HTML autonome qui guide le montage de `sequence` sur `toile`."""
    seq = [int(c) for c in sequence]
    if len(seq) < 2:
        raise ValueError("séquence trop courte pour un guide (au moins deux clous)")
    # Empreinte de la pièce : deux pièces différentes ne partagent pas leur avancement mémorisé.
    empreinte = 0
    for c in seq:
        empreinte = (empreinte * 131 + c + 1) % 4_294_967_291
    donnees = {
        "sequence": seq,
        "clous": toile.n_clous,
        "diametre_mm": toile.diametre_mm,
        "fil_mm": toile.epaisseur_fil_mm,
        "fil_m": round(longueur_fil_m(toile, seq)),
        "empreinte": f"{toile.n_clous}-{len(seq)}-{empreinte}",
    }
    return _GABARIT.replace("__DONNEES__", json.dumps(donnees, separators=(",", ":")))
