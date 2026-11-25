"""
Fig. 1 - causal-loop diagram of the system-dynamics model (nation_state.mdl).

Run (from implementation/):  .venv/bin/python causal_loops.py
Output: ../manuscript/figures/F1_causal_loops.png|pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

FIG = Path(__file__).resolve().parent.parent / "manuscript" / "figures"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
POS_C, NEG_C = "#2a78d6", "#eb6834"  # + links blue, - links orange (sign also written)

NODES = {
    "C": ((0.03, 0.50), "AI\ncapability"),
    "ae": ((0.30, 0.86), "Economic\nautomation $a_e$"),
    "af": ((0.30, 0.14), "Force\nautomation $a_f$"),
    "X": ((0.30, 0.50), "Contract\nbroken"),
    "B": ((0.66, 0.50), "Citizens'\nbargaining\npower $B$"),
    "L": ((0.97, 0.86), "Legitimacy\n$L_g$"),
    "P": ((0.97, 0.14), "Power\nconcentration\n$P$"),
}
# (from, to, sign, curvature, optional link label)
EDGES = [
    ("C", "ae", "+", 0.0, None), ("C", "af", "+", 0.0, None),
    ("ae", "B", "−", -0.18, None), ("B", "ae", "−", -0.18, "brake $k\\cdot B$"),
    ("af", "B", "−", 0.18, None), ("B", "af", "−", 0.18, "brake $k\\cdot B$"),
    ("ae", "X", "+", 0.0, None), ("af", "X", "+", 0.0, None),
    ("X", "B", "−", 0.0, "repression"),
    ("B", "L", "+", 0.0, None), ("X", "L", "−", -0.42, None),
    ("B", "P", "−", 0.0, None), ("L", "P", "−", 0.0, None),
    ("P", "af", "+", 0.0, "coup-proofing"),
]
LOOPS = [((0.49, 0.69), "R1"), ((0.49, 0.31), "R1"),
         ((0.75, 0.27), "R2"), ((0.44, 0.555), "R3")]


def bezier(p1, p2, rad, t):
    """Point on matplotlib's arc3 quadratic curve at parameter t."""
    (x1, y1), (x2, y2) = p1, p2
    cx, cy = (x1 + x2) / 2 + rad * (y2 - y1), (y1 + y2) / 2 - rad * (x2 - x1)
    x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2
    y = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2
    return x, y


def main():
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.set_xlim(-0.05, 1.09)
    ax.set_ylim(-0.04, 1.0)
    ax.axis("off")
    for _, ((x, y), lbl) in NODES.items():
        ax.text(x, y, lbl, ha="center", va="center", fontsize=9, color=INK,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=MUTED, lw=0.8), zorder=3)
    for a, b, sign, rad, lbl in EDGES:
        p1, p2 = NODES[a][0], NODES[b][0]
        col = POS_C if sign == "+" else NEG_C
        ax.add_patch(FancyArrowPatch(p1, p2, connectionstyle=f"arc3,rad={rad}",
                                     arrowstyle="-|>", mutation_scale=12, lw=1.4, color=col,
                                     shrinkA=32, shrinkB=36, zorder=2))
        # sign placed beside the curve (offset perpendicular to it), near the head
        sx, sy = bezier(p1, p2, rad, 0.70)
        tx, ty = bezier(p1, p2, rad, 0.72)
        nx, ny = -(ty - sy), (tx - sx)
        norm = max((nx ** 2 + ny ** 2) ** 0.5, 1e-9)
        ax.text(sx + 0.03 * nx / norm, sy + 0.03 * ny / norm, sign, fontsize=12, color=col,
                weight="bold", ha="center", va="center", zorder=4)
        if lbl:
            mx, my = bezier(p1, p2, rad, 0.5)
            ax.text(mx, my, lbl, fontsize=7.5, color=INK2, ha="center", va="center", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    for (x, y), lbl in LOOPS:
        ax.text(x, y, lbl, ha="center", va="center", fontsize=8.5, color=INK2, style="italic")
    ax.text(-0.03, 0.0, "Blue +: both move in the same direction.  Orange −: opposite directions.  "
            "R1 erosion, R2 coup-proofing, R3 repression lock-in (all reinforcing).\n"
            "Every stock also adjusts towards its target with a delay (balancing; not drawn).",
            fontsize=7.5, color=INK2)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"F1_causal_loops.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
