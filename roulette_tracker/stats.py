"""
stats.py — Statistiques et visualisations des résultats de roulette.
"""

from collections import Counter
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")   # pas besoin d'écran
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from detector import RouletteResult, RED_NUMBERS


# ── Couleur de barre pour chaque numéro ──────────────────────────────────────

def _bar_color(n: int) -> str:
    if n == 0:
        return "#2ecc71"   # vert
    return "#e74c3c" if n in RED_NUMBERS else "#2c3e50"


# ── Résumé texte ──────────────────────────────────────────────────────────────

def print_summary(results: List[RouletteResult]):
    if not results:
        print("Aucun résultat.")
        return

    total = len(results)
    numbers = [r.number for r in results]
    counts  = Counter(numbers)

    print("\n" + "═" * 55)
    print(f"  RÉSUMÉ — {total} tours")
    print("═" * 55)

    # Couleurs
    colors = Counter(r.color for r in results)
    print(f"\n  Couleurs")
    for color, count in sorted(colors.items()):
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {color:<8} {count:>4}  {pct:>5.1f}%  {bar}")

    # Parités
    parities = Counter(r.parity for r in results)
    print(f"\n  Parités")
    for p, count in sorted(parities.items()):
        pct = count / total * 100
        print(f"    {p:<8} {count:>4}  {pct:>5.1f}%")

    # Douzaines
    dozens = Counter(r.dozen for r in results)
    print(f"\n  Douzaines")
    for d, count in sorted(dozens.items()):
        pct = count / total * 100
        print(f"    {d:<8} {count:>4}  {pct:>5.1f}%")

    # Top 5 / Bottom 5
    print(f"\n  Top 5 numéros les plus fréquents")
    for n, c in counts.most_common(5):
        pct = c / total * 100
        print(f"    {n:>2}  →  {c}x  ({pct:.1f}%)")

    print(f"\n  Top 5 numéros les moins fréquents")
    for n, c in counts.most_common()[:-6:-1]:
        pct = c / total * 100
        print(f"    {n:>2}  →  {c}x  ({pct:.1f}%)")

    # Numéros jamais sortis
    seen     = set(numbers)
    missing  = sorted(set(range(37)) - seen)
    if missing:
        print(f"\n  Numéros jamais sortis ({len(missing)}) : {missing}")

    print("═" * 55 + "\n")


# ── Graphiques ────────────────────────────────────────────────────────────────

def plot_frequency(results: List[RouletteResult], output_dir: str | Path) -> Path:
    """Histogramme de fréquence par numéro (0–36)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    numbers = [r.number for r in results]
    counts  = Counter(numbers)
    x       = list(range(37))
    y       = [counts.get(n, 0) for n in x]
    colors  = [_bar_color(n) for n in x]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x, y, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Numéro")
    ax.set_ylabel("Fréquence")
    ax.set_title(f"Fréquence de sortie par numéro ({len(results)} tours)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in x], fontsize=8)

    # Légende
    legend = [
        mpatches.Patch(color="#2ecc71", label="0 (vert)"),
        mpatches.Patch(color="#e74c3c", label="Rouge"),
        mpatches.Patch(color="#2c3e50", label="Noir"),
    ]
    ax.legend(handles=legend, loc="upper right")
    ax.axhline(len(results) / 37, color="gray", linestyle="--", linewidth=1,
               label="Espérance théorique")

    plt.tight_layout()
    out = output_dir / "frequency.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Stats] frequency.png → {out}")
    return out


def plot_color_pie(results: List[RouletteResult], output_dir: str | Path) -> Path:
    """Camembert rouge / noir / vert."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    colors  = Counter(r.color for r in results)
    labels  = list(colors.keys())
    values  = list(colors.values())
    palette = {"red": "#e74c3c", "black": "#2c3e50", "green": "#2ecc71"}
    clrs    = [palette.get(l, "#aaa") for l in labels]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=clrs,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    ax.set_title(f"Répartition couleurs ({len(results)} tours)")
    plt.tight_layout()
    out = output_dir / "color_pie.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Stats] color_pie.png → {out}")
    return out


def plot_timeline(results: List[RouletteResult], output_dir: str | Path) -> Path:
    """Chronologie des numéros sortis au fil du temps."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    turns   = list(range(1, len(results) + 1))
    numbers = [r.number for r in results]
    colors  = [_bar_color(r.number) for r in results]

    fig, ax = plt.subplots(figsize=(max(10, len(results) * 0.4), 5))
    ax.scatter(turns, numbers, c=colors, s=80, zorder=3)
    ax.plot(turns, numbers, color="gray", linewidth=0.5, zorder=2)
    ax.set_xlabel("Tour n°")
    ax.set_ylabel("Numéro sorti")
    ax.set_title("Chronologie des résultats")
    ax.set_yticks(range(0, 37, 3))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = output_dir / "timeline.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Stats] timeline.png → {out}")
    return out


def plot_heatmap(results: List[RouletteResult], output_dir: str | Path) -> Path:
    """Heatmap de fréquence sur une représentation simplifiée de la table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Table roulette européenne : 3 colonnes × 12 lignes + 0
    counts = Counter(r.number for r in results)
    # Grille 3×12 pour 1–36
    grid = np.zeros((12, 3))
    for n in range(1, 37):
        row = (n - 1) // 3
        col = (n - 1) % 3
        grid[row, col] = counts.get(n, 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8),
                                    gridspec_kw={"width_ratios": [3, 1]})

    # Grille principale 1–36
    im = ax1.imshow(grid, cmap="YlOrRd", aspect="auto")
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(["Col 1", "Col 2", "Col 3"])
    ax1.set_yticks(range(12))
    ax1.set_yticklabels([f"{r*3+1}–{r*3+3}" for r in range(12)])
    ax1.set_title("Heatmap 1–36")
    for n in range(1, 37):
        row = (n - 1) // 3
        col = (n - 1) % 3
        ax1.text(col, row, f"{n}\n({counts.get(n, 0)})",
                 ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax1, label="Fréquence")

    # 0 isolé
    ax2.bar(["0"], [counts.get(0, 0)], color="#2ecc71")
    ax2.set_title(f"0 → {counts.get(0, 0)}x")
    ax2.set_ylabel("Fréquence")

    plt.suptitle(f"Heatmap des numéros ({len(results)} tours)", y=1.01)
    plt.tight_layout()
    out = output_dir / "heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Stats] heatmap.png → {out}")
    return out


def generate_all(results: List[RouletteResult], output_dir: str | Path):
    """Génère tous les graphiques + résumé texte."""
    print_summary(results)
    plot_frequency(results, output_dir)
    plot_color_pie(results, output_dir)
    plot_timeline(results, output_dir)
    plot_heatmap(results, output_dir)
    print(f"[Stats] Tous les graphiques dans : {Path(output_dir).resolve()}")
