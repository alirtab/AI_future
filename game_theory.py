"""
Step 2 - Game-theoretic models for
"When the State No Longer Needs the Nation: AI and the Future of the Social Contract".

Game A - Automation race between two rival states (2x2 simultaneous game).
Game B - Ruler vs. citizens under automation (sequential game, solved by backward induction).

All parameter values below are ILLUSTRATIVE and must be approved by the author.
They are normalised (the ruler's total rent R = 1) and carry no empirical claim.

Run:  .venv/bin/python game_theory.py
Output: figures/*.png|pdf, plus a text summary on stdout.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parent
FIG = (PAPER_ROOT / "manuscript" / "figures"
       if (PAPER_ROOT / "manuscript").is_dir() else HERE / "figures")
FIG.mkdir(exist_ok=True)

# Colours: validated categorical palette (slots 1-4) + neutral ink; hatching for greyscale print.
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9"}
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": C["ink"],
    "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    "lines.linewidth": 2,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


# =====================================================================================
# GAME A - Automation race
# =====================================================================================
# Players: two rival states. Strategies: F = automate state functions fast, S = slow.
# Parameters (all >= 0):
#   b  advantage gained by automating fast while the rival is slow
#   c  loss from falling behind when the rival automates fast and you do not
#   k  private domestic cost of fast automation (legitimacy loss, disruption);
#      we hypothesise k is higher in accountable regimes than in autocracies
#   h  shared expected harm each fast automator adds for BOTH players
#      (accident / loss of control / destabilisation)
# Payoffs to the row player (baseline S,S = 0):
#   (S,S): 0            (F,S): b - k - h
#   (S,F): -c - h       (F,F): -k - 2h
# Standard labels: R = (S,S), T = (F,S), S* = (S,F), P = (F,F).
# Classification depends only on b/(k+h) and c/(k+h):
#   PD            : b > k+h and c > k+h   (F dominant, (F,F) Pareto-worse than (S,S))
#   Stag hunt     : b < k+h and c > k+h   (two pure equilibria; coordination problem)
#   Chicken       : b > k+h and c < k+h   (anti-coordination)
#   Harmony       : b < k+h and c < k+h   (S dominant; no dilemma)

def game_a_payoffs(b, c, k, h):
    R, T, S_, P = 0.0, b - k - h, -c - h, -k - 2 * h
    return R, T, S_, P


def classify(b, c, k, h):
    t = k + h
    if b > t and c > t:
        return "Prisoner's dilemma"
    if b < t and c > t:
        return "Stag hunt"
    if b > t and c < t:
        return "Chicken"
    return "Harmony"


REGIONS = {  # name: (colour, hatch)
    "Harmony": (C["aqua"], ""),
    "Stag hunt": (C["yellow"], ".."),
    "Chicken": (C["blue"], "\\\\"),
    "Prisoner's dilemma": (C["orange"], "//"),
}


def fig_game_a_regions():
    """Region map in normalised coordinates x = b/(k+h), y = c/(k+h)."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    m = 3.0
    boxes = {"Harmony": (0, 0, 1, 1), "Stag hunt": (0, 1, 1, m - 1),
             "Chicken": (1, 0, m - 1, 1), "Prisoner's dilemma": (1, 1, m - 1, m - 1)}
    for name, (x, y, w, hgt) in boxes.items():
        col, hatch = REGIONS[name]
        ax.add_patch(plt.Rectangle((x, y), w, hgt, facecolor=col, alpha=0.28,
                                   hatch=hatch, edgecolor=col, linewidth=0))
        ax.text(x + w / 2, y + hgt / 2, name, ha="center", va="center",
                color=C["ink"], fontsize=10, weight="bold")
    ax.axvline(1, color=C["ink2"], lw=1)
    ax.axhline(1, color=C["ink2"], lw=1)
    # Illustrative regime points (same b, c; different domestic cost k)
    for lbl, k, mk in (("Accountable regime (k=0.7)", 0.7, "o"),
                       ("Autocracy (k=0.1)", 0.1, "s")):
        b, c, h = 0.8, 0.7, 0.2
        ax.plot(b / (k + h), c / (k + h), mk, ms=9, color=C["ink"],
                markeredgecolor="white", markeredgewidth=2, label=lbl)
    ax.set_xlim(0, m)
    ax.set_ylim(0, m)
    ax.set_xlabel("Gain from moving first,  b / (k + h)")
    ax.set_ylabel("Fear of falling behind,  c / (k + h)")
    ax.set_title("Game A: what kind of game is the automation race?",
                 loc="left", fontsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=8, bbox_to_anchor=(0, -0.13))
    save(fig, "A1_regions")


def fig_game_a_matrices():
    """Two payoff matrices: the same race in an accountable regime vs. an autocracy."""
    cases = [("Accountable regime", 0.8, 0.7, 0.7, 0.2),
             ("Autocracy", 0.8, 0.7, 0.1, 0.2)]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, (title, b, c, k, h) in zip(axes, cases):
        R, T, S_, P = game_a_payoffs(b, c, k, h)
        cell = {("S", "S"): (R, R), ("S", "F"): (S_, T),
                ("F", "S"): (T, S_), ("F", "F"): (P, P)}
        # pure Nash equilibria
        def br_row(col):
            return max(("S", "F"), key=lambda r: cell[(r, col)][0])
        def br_col(row):
            return max(("S", "F"), key=lambda cc: cell[(row, cc)][1])
        nash = [(r, cc) for r in "SF" for cc in "SF"
                if br_row(cc) == r and br_col(r) == cc]
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.invert_yaxis()
        ax.set_xticks([0.5, 1.5], ["Slow", "Fast"])
        ax.xaxis.tick_top()
        ax.set_yticks([0.5, 1.5], ["Slow", "Fast"])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        for i, r in enumerate("SF"):
            for j, cc in enumerate("SF"):
                is_ne = (r, cc) in nash
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=C["orange"] if is_ne else "white",
                                           alpha=0.25 if is_ne else 1, edgecolor=C["ink2"]))
                u1, u2 = cell[(r, cc)]
                ax.text(j + 0.5, i + 0.5, f"{u1:+.2f} , {u2:+.2f}" + ("\nNash eq." if is_ne else ""),
                        ha="center", va="center", fontsize=10, color=C["ink"])
        ax.set_title(f"{title}  (k = {k})\n{classify(b, c, k, h)}", fontsize=10, pad=24)
    fig.text(0.5, -0.04, "Rows: State 1, columns: State 2. Payoffs (State 1, State 2). "
             "b = 0.8, c = 0.7, h = 0.2. Shaded = pure-strategy Nash equilibrium.",
             ha="center", fontsize=8, color=C["ink2"])
    save(fig, "A2_matrices")


def fig_game_a_replicator():
    """Replicator dynamics in a population of states: dx/dt = x(1-x)(pi_F - pi_S)."""
    x = np.linspace(0, 1, 401)
    cases = [("Prisoner's dilemma", 0.8, 0.7, 0.1, 0.2),
             ("Stag hunt", 0.1, 0.7, 0.1, 0.2),
             ("Harmony", 0.3, 0.2, 0.3, 0.2)]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for name, b, c, k, h in cases:
        R, T, S_, P = game_a_payoffs(b, c, k, h)
        pi_f = x * P + (1 - x) * T
        pi_s = x * S_ + (1 - x) * R
        ax.plot(x, x * (1 - x) * (pi_f - pi_s), color=REGIONS[name][0], label=name)
    ax.axhline(0, color=C["ink2"], lw=1)
    ax.set_xlabel("Share of states automating fast, x")
    ax.set_ylabel("Rate of change, dx/dt")
    ax.set_title("Game A: evolutionary dynamics (above 0 = more states race)",
                 loc="left", fontsize=11)
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "A3_replicator")


# =====================================================================================
# GAME B - Ruler vs. citizens under automation
# =====================================================================================
# Sequence: (1) Ruler chooses SHARE (concede share s of rent) or REPRESS.
#           (2) If REPRESS, citizens choose COMPLY or RESIST.
# Two automation indices in [0,1]:
#   a_e = automated share of the ECONOMY (production + administration)
#   a_f = automated share of FORCE (police, military, surveillance)
# Mechanisms (each a modelling assumption, approved by the author):
#   - Dependence (economy): under repression, uncooperative citizens cut the ruler's rent
#                   by theta*(1-a_e). With a fully automated economy the ruler no longer needs them.
#   - Repression cost (force): m(a_f) = m0*(1-a_f); robots are cheaper than soldiers.
#   - Refusal veto (force): resistance succeeds with probability pi(a_f) = pi0*(1-a_f)**gamma,
#                   because soldiers can defect but robots cannot.
# Payoffs (ruler, citizens), with keep = R(1-theta(1-a_e)) - m(a_f):
#   SHARE                : ((1-s)R,  sR)
#   REPRESS, COMPLY      : (keep,  0)
#   REPRESS, RESIST win  : (-L,  R - kappa)        prob pi(a_f)
#   REPRESS, RESIST lose : (keep,  -d)             prob 1-pi(a_f)
# Figures B2/B3 use balanced automation a_e = a_f = a; figure B4 separates them.

B_PARAMS = dict(R=1.0, s=0.3, theta=0.5, m0=0.2, L=1.0, d=0.5, kappa=0.0, pi0=0.6, gamma=1.0)


def game_b_solve(a, R, s, theta, m0, L, d, kappa, pi0, gamma, a_f=None):
    """a = economic automation a_e; a_f = force automation (defaults to a: balanced)."""
    a_f = a if a_f is None else a_f
    pi = pi0 * (1 - a_f) ** gamma
    keep = R * (1 - theta * (1 - a)) - m0 * (1 - a_f)
    resist_c = pi * (R - kappa) + (1 - pi) * (-d)
    citizens_resist = resist_c > 0
    repress_r = (pi * (-L) + (1 - pi) * keep) if citizens_resist else keep
    share_r = (1 - s) * R
    ruler_represses = repress_r > share_r
    return dict(pi=pi, keep=keep, repress=repress_r, share=share_r,
                citizens_resist=citizens_resist, ruler_represses=ruler_represses)


def thresholds(params, n=10001):
    a = np.linspace(0, 1, n)
    sol = [game_b_solve(ai, **params) for ai in a]
    a_c = next((ai for ai, s_ in zip(a, sol) if not s_["citizens_resist"]), None)
    a_r = next((ai for ai, s_ in zip(a, sol) if s_["ruler_represses"]), None)
    return a, sol, a_c, a_r


def fig_game_b_tree():
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.axis("off")
    nodes = {"R": (0.1, 0.5), "share": (0.45, 0.85), "C": (0.45, 0.25),
             "comply": (0.85, 0.45), "resist": (0.85, 0.05)}
    edges = [("R", "share", "Share power (s)"), ("R", "C", "Repress"),
             ("C", "comply", "Comply"), ("C", "resist", "Resist")]
    for u, v, lbl in edges:
        (x1, y1), (x2, y2) = nodes[u], nodes[v]
        ax.plot([x1, x2], [y1, y2], color=C["ink2"], lw=1.5)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.04, lbl, ha="center", fontsize=9, color=C["ink"])
    for key, txt in (("R", "Ruler"), ("C", "Citizens")):
        x, y = nodes[key]
        ax.plot(x, y, "o", ms=12, color=C["blue"], markeredgecolor="white", markeredgewidth=2)
        ax.text(x, y - 0.09, txt, ha="center", fontsize=9, weight="bold")
    leaves = {"share": "((1-s)R ,  sR)",
              "comply": "(R[1-θ(1-a)] - m(a) ,  0)",
              "resist": "win, p = π(a):   (-L ,  R - κ)\nlose, 1 - π(a):  (R[1-θ(1-a)] - m(a) ,  -d)"}
    for key, txt in leaves.items():
        x, y = nodes[key]
        ax.plot(x, y, "s", ms=7, color=C["ink"])
        ax.text(x + 0.02, y, txt, va="center", fontsize=8.5, family="monospace")
    ax.text(0.0, -0.12, "a = automated share of state functions;  π(a) = π0(1-a)^γ (the soldiers' "
            "refusal veto);  m(a) = m0(1-a) (repression cost);  θ = ruler's dependence on citizens' "
            "cooperation.", fontsize=7.5, color=C["ink2"], transform=ax.transAxes)
    ax.set_xlim(0, 1.25)
    ax.set_ylim(-0.05, 1)
    ax.set_title("Game B: ruler vs. citizens (payoffs: ruler, citizens)", loc="left", fontsize=11)
    save(fig, "B1_tree")


def fig_game_b_tipping():
    a, sol, a_c, a_r = thresholds(B_PARAMS)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(a, [s_["share"] for s_ in sol], color=C["blue"], label="Ruler payoff: share power")
    ax.plot(a, [s_["repress"] for s_ in sol], color=C["orange"], label="Ruler payoff: repress")
    if a_c is not None:
        ax.axvline(a_c, color=C["muted"], lw=1, ls="--")
        ax.text(a_c - 0.01, 0.9, f"a_c = {a_c:.2f}\ncitizens stop resisting",
                fontsize=8, color=C["ink2"], ha="right")
    if a_r is not None:
        ax.axvline(a_r, color=C["ink"], lw=1)
        ax.text(a_r + 0.01, 0.15, f"a* = {a_r:.2f}\nrepression becomes\nthe ruler's best reply",
                fontsize=8, color=C["ink"])
    ax.set_xlabel("Automated share of state functions, a (balanced: a_e = a_f)")
    ax.set_ylabel("Ruler's expected payoff (R = 1)")
    ax.set_title("Game B: the tipping point of the social contract", loc="left", fontsize=11)
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, "B2_tipping")
    return a_c, a_r


def fig_game_b_heatmap():
    """Equilibrium outcome over automation a and dependence theta."""
    a_vals = np.linspace(0, 1, 201)
    th_vals = np.linspace(0, 1, 201)
    Z = np.zeros((len(th_vals), len(a_vals)))
    for i, th in enumerate(th_vals):
        p = dict(B_PARAMS, theta=th)
        for j, ai in enumerate(a_vals):
            s_ = game_b_solve(ai, **p)
            Z[i, j] = 0 if not s_["ruler_represses"] else (1 if s_["citizens_resist"] else 2)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([C["blue"], C["yellow"], C["orange"]])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.imshow(Z, origin="lower", extent=(0, 1, 0, 1), aspect="auto", cmap=cmap,
              vmin=0, vmax=2, alpha=0.75, interpolation="nearest")
    ax.set_xlabel("Automated share of state functions, a (balanced: a_e = a_f)")
    ax.set_ylabel("Ruler's dependence on citizens, θ")
    ax.set_title("Game B: equilibrium outcome", loc="left", fontsize=11)
    labels = [(0, C["blue"], "Contract holds (ruler shares)"),
              (1, C["yellow"], "Repression, citizens resist"),
              (2, C["orange"], "Repression, citizens comply")]
    ax.legend(handles=[Patch(color=col, alpha=0.75, label=lbl)
                       for v, col, lbl in labels if (Z == v).any()],
              loc="upper left", bbox_to_anchor=(0, -0.14), frameon=False, fontsize=8)
    save(fig, "B3_heatmap")


def fig_game_b_split():
    """Equilibrium over economic automation a_e (x) and force automation a_f (y)."""
    from matplotlib.colors import ListedColormap
    n = 301
    ae = np.linspace(0, 1, n)
    af = np.linspace(0, 1, n)
    Z = np.zeros((n, n))
    for i, f in enumerate(af):
        for j, e in enumerate(ae):
            s_ = game_b_solve(e, a_f=f, **B_PARAMS)
            Z[i, j] = 0 if not s_["ruler_represses"] else (1 if s_["citizens_resist"] else 2)
    cmap = ListedColormap([C["blue"], C["yellow"], C["orange"]])
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.imshow(Z, origin="lower", extent=(0, 1, 0, 1), aspect="auto", cmap=cmap,
              vmin=0, vmax=2, alpha=0.75, interpolation="nearest")
    ax.plot([0, 1], [0, 1], color=C["ink2"], lw=1, ls="--")
    ax.text(0.05, 0.30, "balanced path\n(a_e = a_f)", fontsize=8, color=C["ink2"], rotation=0)
    ax.set_xlabel("Economic automation, a_e (production + administration)")
    ax.set_ylabel("Force automation, a_f (police + military)")
    ax.set_title("Game B: which automation breaks the contract?", loc="left", fontsize=11)
    labels = [(0, C["blue"], "Contract holds (ruler shares)"),
              (1, C["yellow"], "Repression, citizens resist"),
              (2, C["orange"], "Repression, citizens comply")]
    ax.legend(handles=[Patch(color=col, alpha=0.75, label=lbl)
                       for v, col, lbl in labels if (Z == v).any()],
              loc="upper left", bbox_to_anchor=(0, -0.14), frameon=False, fontsize=8)
    save(fig, "B4_force_vs_economy")
    # report: minimum a_f needed for repression at a_e = 0, and min a_e needed at a_f = 0
    rep = Z > 0
    col0 = af[np.argmax(rep[:, 0])] if rep[:, 0].any() else None
    row0 = ae[np.argmax(rep[0, :])] if rep[0, :].any() else None
    return col0, row0


def sensitivity_b():
    """How a* moves when each parameter is varied +/- 50 %."""
    base = thresholds(B_PARAMS)[3]
    rows = []
    for key in ("s", "theta", "m0", "L", "d", "pi0"):
        out = []
        for f in (0.5, 1.5):
            p = dict(B_PARAMS, **{key: B_PARAMS[key] * f})
            out.append(thresholds(p, n=2001)[3])
        rows.append((key, out[0], out[1]))
    return base, rows


if __name__ == "__main__":
    fig_game_a_regions()
    fig_game_a_matrices()
    fig_game_a_replicator()
    fig_game_b_tree()
    a_c, a_r = fig_game_b_tipping()
    fig_game_b_heatmap()
    af_only, ae_only = fig_game_b_split()

    print("GAME A (b=0.8, c=0.7, h=0.2)")
    for k in (0.7, 0.1):
        print(f"  k={k}: {classify(0.8, 0.7, k, 0.2)}  payoffs R,T,S,P = "
              f"{tuple(round(v, 2) for v in game_a_payoffs(0.8, 0.7, k, 0.2))}")
    print("GAME B", B_PARAMS)
    print(f"  citizens stop resisting at a_c = {a_c:.3f}")
    print(f"  ruler prefers repression at a* = {a_r:.3f}")
    print(f"  split model: with a_e = 0, repression needs a_f >= {af_only}")
    print(f"  split model: with a_f = 0, repression needs a_e >= {ae_only}")
    base, rows = sensitivity_b()
    print("  sensitivity of a* (param x0.5 , x1.5):")
    for key, lo, hi in rows:
        fmt = lambda v: "none" if v is None else f"{v:.3f}"
        print(f"    {key:6s}: {fmt(lo)} , {fmt(hi)}")
    print(f"Figures written to {FIG}")
