"""
Step 3 - System-dynamics scenarios and robustness for
"When the State No Longer Needs the Nation".

Single source of truth: nation_state.mdl (Vensim). This script runs it with PySD.

Run:  .venv/bin/python sd_runs.py           (everything)
     .venv/bin/python sd_runs.py --no-mc   (skip Monte Carlo)
Output: figures/S*.png|pdf, results/*.csv, summary on stdout.
"""

import sys
import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parent
ROOT = PAPER_ROOT if (PAPER_ROOT / "manuscript").is_dir() else HERE
FIG = (ROOT / "manuscript" / "figures"
       if (ROOT / "manuscript").is_dir() else ROOT / "figures")
RES = ROOT / "implementation" / "results" if ROOT != HERE else ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
from game_theory import C, B_PARAMS, game_b_solve, save  # noqa: E402  (shared style + Game B)

MODEL = pysd.read_vensim(str(HERE / "nation_state.mdl"))

REGIMES = {"Accountable": 0.7, "Autocracy": 0.1}
SPEEDS = {"Slow": 0.08, "Current": 0.15, "Fast": 0.30}
POLICIES = {"None": (0.0, 0.0), "AI diffusion": (0.5, 0.0),
            "Distributed force": (0.0, 0.5), "Both": (0.5, 0.5)}
SPEED_COL = {"Slow": C["aqua"], "Current": C["blue"], "Fast": C["orange"]}
OUT = ["AI Capability", "Economic Automation", "Force Automation", "Bargaining Power",
       "Legitimacy", "Power Concentration", "Contract Broken",
       "Effective Economic Automation", "Effective Force Automation"]


def params(regime, speed, policy, **extra):
    sigma, D = POLICIES[policy]
    p = {"Accountability": REGIMES[regime], "AI Speed": SPEEDS[speed],
         "AI Diffusion Policy": sigma, "Distributed Force Control": D}
    p.update(extra)
    return p


def full(p):
    """PySD keeps parameter changes between runs, so always pass every constant."""
    return {**DEFAULTS, **p}


START = 2025  # results are reported from here; 1975-2025 is the model's spin-up period


def run(p, final_time=2075):
    return MODEL.run(params=full(p), final_time=final_time, return_columns=OUT).loc[START:]


def break_year(df):
    df = df.loc[START:]
    hit = df.index[df["Contract Broken"] > 0.5]
    return float(hit.min()) if len(hit) else np.inf


def fmt_year(y):
    return "holds" if np.isinf(y) else f"{y:.0f}"


# ------------------------------------------------------------------ 24 scenarios
def scenarios():
    rows, series = [], {}
    for (rg, sp, po) in product(REGIMES, SPEEDS, POLICIES):
        df = run(params(rg, sp, po))
        series[(rg, sp, po)] = df
        last = df.iloc[-1]
        rows.append(dict(regime=rg, speed=sp, policy=po, break_year=break_year(df),
                         bargaining_2075=last["Bargaining Power"],
                         legitimacy_2075=last["Legitimacy"],
                         concentration_2075=last["Power Concentration"]))
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "scenarios.csv", index=False)
    return tab, series


def fig_s1(series):
    """Small multiples: bargaining power over time; rows = regime, cols = policy, lines = speed."""
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), sharex=True, sharey=True)
    for i, rg in enumerate(REGIMES):
        for j, po in enumerate(POLICIES):
            ax = axes[i, j]
            for sp in SPEEDS:
                df = series[(rg, sp, po)]
                ax.plot(df.index, df["Bargaining Power"], color=SPEED_COL[sp], label=f"{sp} AI")
                by = break_year(df)
                if not np.isinf(by):
                    ax.plot(by, df.loc[by, "Bargaining Power"], "v", ms=8, color=SPEED_COL[sp],
                            markeredgecolor="white", markeredgewidth=1.5)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", color=C["grid"], lw=0.6)
            if i == 0:
                ax.set_title(f"Policy: {po}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{rg}\nCitizens' bargaining power")
            if i == 1:
                ax.set_xlabel("Year")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("System dynamics: citizens' bargaining power, 2025–2075 "
                 "(▼ = year the social contract breaks)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "S1_bargaining_power")


def fig_s2(tab):
    """Break year for all 24 scenarios as a coloured table."""
    rows = [(rg, po) for rg in REGIMES for po in POLICIES]
    look = tab.set_index(["regime", "policy", "speed"])["break_year"]
    Z = np.array([[look[(rg, po, sp)] for sp in SPEEDS] for rg, po in rows])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    holds = np.isinf(Z)
    ax.imshow(np.where(holds, 0, 1), cmap=ListedColormap([C["blue"], C["orange"]]),
              vmin=0, vmax=1, alpha=0.35, aspect="auto")
    for (r, c), v in np.ndenumerate(Z):
        ax.text(c, r, "holds" if np.isinf(v) else f"breaks\n{v:.0f}", ha="center", va="center",
                fontsize=9, color=C["ink"])
    ax.set_xticks(range(len(SPEEDS)), [f"{s} AI" for s in SPEEDS])
    ax.set_yticks(range(len(rows)), [f"{rg} · {po}" for rg, po in rows])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(3.5, color=C["ink2"], lw=1)
    ax.set_title("Does the social contract survive to 2075?", loc="left", fontsize=11, pad=28)
    save(fig, "S2_break_years")


def fig_s3(series):
    """Trajectories in the (effective a_e, effective a_f) plane over the Game B regions."""
    n = 301
    g = np.linspace(0, 1, n)
    Z = np.array([[game_b_solve(e, a_f=f, **B_PARAMS)["ruler_represses"] for e in g] for f in g])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(Z, origin="lower", extent=(0, 1, 0, 1), aspect="auto",
              cmap=ListedColormap([C["blue"], C["orange"]]), alpha=0.18, interpolation="nearest")
    cases = [("Autocracy", "Current", "None", C["ink"], "-"),
             ("Autocracy", "Current", "AI diffusion", C["aqua"], "-"),
             ("Autocracy", "Current", "Distributed force", C["yellow"], "-"),
             ("Autocracy", "Current", "Both", C["blue"], "-"),
             ("Accountable", "Fast", "None", C["ink"], "--")]
    for rg, sp, po, col, ls in cases:
        df = series[(rg, sp, po)]
        x, y = df["Effective Economic Automation"], df["Effective Force Automation"]
        ax.plot(x, y, color=col, ls=ls, label=f"{rg}, {sp.lower()} AI, policy: {po}")
        ax.plot(x.iloc[-1], y.iloc[-1], "o", ms=7, color=col, markeredgecolor="white",
                markeredgewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Effective economic automation (machines that replace citizens)")
    ax.set_ylabel("Effective force automation (machines that obey the ruler alone)")
    ax.set_title("Paths of five scenarios across the Game B map, 2025 → 2075 (●)",
                 loc="left", fontsize=11)
    h, l_ = ax.get_legend_handles_labels()
    h += [Patch(color=C["blue"], alpha=0.3), Patch(color=C["orange"], alpha=0.3)]
    l_ += ["Game B: contract holds", "Game B: contract broken"]
    ax.legend(h, l_, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.12))
    save(fig, "S3_phase_paths")


# ------------------------------------------------------------------ coup-proofing loop
def coup_proofing():
    rows = []
    for rg, sp in product(REGIMES, SPEEDS):
        on = run(params(rg, sp, "None"))
        off = run(params(rg, sp, "None", **{"Coup Proofing Switch": 0}))
        rows.append(dict(regime=rg, speed=sp, break_with_loop=break_year(on),
                         break_without_loop=break_year(off),
                         concentration_2075_with=on["Power Concentration"].iloc[-1],
                         concentration_2075_without=off["Power Concentration"].iloc[-1]))
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "coup_proofing.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (rg, sp) in zip(axes, (("Accountable", "Fast"), ("Autocracy", "Current"))):
        for sw, ls, lbl in ((1, "-", "with coup-proofing loop (R2)"),
                            (0, "--", "without loop")):
            df = run(params(rg, sp, "None", **{"Coup Proofing Switch": sw}))
            ax.plot(df.index, df["Force Automation"], color=C["orange"], ls=ls,
                    label=f"Force automation, {lbl}")
            ax.plot(df.index, df["Bargaining Power"], color=C["blue"], ls=ls,
                    label=f"Bargaining power, {lbl}")
            by = break_year(df)
            if not np.isinf(by):
                ax.axvline(by, color=C["muted"], ls=ls, lw=1)
        ax.set_title(f"{rg}, {sp.lower()} AI, no policy", fontsize=10, loc="left")
        ax.set_xlabel("Year")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color=C["grid"], lw=0.6)
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2)
    fig.suptitle("Sensitivity: removing the coup-proofing loop (grey lines = contract breaks)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "S4_coup_proofing")
    return tab


# ------------------------------------------------------------------ one-at-a-time +/-50 %
OAT = {"Economic Automation Delay": 5, "Force Automation Delay": 8, "Bargaining Power Delay": 3,
       "Legitimacy Delay": 5, "Power Concentration Delay": 5, "Dependence on Citizens": 0.5,
       "Citizens Share": 0.3, "Repression Cost": 0.2, "Punishment of Failed Revolt": 0.5,
       "Revolt Success Chance Today": 0.6, "Repression Effect": 0.6, "Economic Weight": 0.5}
DEFAULTS = {**OAT, "Accountability": 0.7, "AI Speed": 0.15, "AI Diffusion Policy": 0.0,
            "Distributed Force Control": 0.0, "Coup Proofing Switch": 1}
CAP1 = {"Dependence on Citizens", "Citizens Share", "Repression Cost",
        "Revolt Success Chance Today", "Repression Effect", "Economic Weight",
        "Punishment of Failed Revolt"}


def tornado():
    bases = (("Accountable", "Current", "None"), ("Autocracy", "Current", "None"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    rows = []
    for ax, (rg, sp, po) in zip(axes, bases):
        base = run(params(rg, sp, po))["Bargaining Power"].iloc[-1]
        res = []
        for k, v in OAT.items():
            lo_v, hi_v = v * 0.5, v * 1.5
            if k in CAP1:
                hi_v = min(hi_v, 0.99)
            lo = run(params(rg, sp, po, **{k: lo_v}))["Bargaining Power"].iloc[-1]
            hi = run(params(rg, sp, po, **{k: hi_v}))["Bargaining Power"].iloc[-1]
            res.append((k, lo, hi))
            rows.append(dict(regime=rg, parameter=k, base=base, low=lo, high=hi))
        res.sort(key=lambda r: abs(r[2] - r[1]))
        y = np.arange(len(res))
        for yi, (k, lo, hi) in zip(y, res):
            ax.barh(yi, lo - base, left=base, color=C["aqua"], height=0.6)
            ax.barh(yi, hi - base, left=base, color="#6250d6",
                    height=0.6)
        ax.axvline(base, color=C["ink"], lw=1)
        ax.set_yticks(y, [r[0] for r in res], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Citizens' bargaining power in 2075")
        ax.set_title(f"{rg}, current AI, no policy (base = {base:.2f})", fontsize=10, loc="left")
    axes[1].legend(handles=[Patch(color=C["aqua"], label="parameter × 0.5"),
                            Patch(color="#6250d6", label="parameter × 1.5")],
                   frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("One-at-a-time sensitivity (±50 %)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "S5_tornado")
    pd.DataFrame(rows).to_csv(RES / "tornado.csv", index=False)


# ------------------------------------------------------------------ Monte Carlo
def monte_carlo(n=1000, seed=2025, horizon=2225):
    """Each draw: every OAT parameter uniform in [0.5x, 1.5x] (fractions capped at 0.99),
    policy strengths uniform in [0.25, 0.75]. Long horizon to see destinations, not timing."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        draw = {k: min(rng.uniform(0.5 * v, 1.5 * v), 0.99) if k in CAP1 else rng.uniform(0.5 * v, 1.5 * v)
                for k, v in OAT.items()}
        sig, D = rng.uniform(0.25, 0.75, 2)
        rec = dict(draw=i, sigma=sig, D=D, **draw)
        for rg in REGIMES:
            def go(sp, s_, d_):
                p = {"Accountability": REGIMES[rg], "AI Speed": SPEEDS[sp],
                     "AI Diffusion Policy": s_, "Distributed Force Control": d_, **draw}
                return break_year(MODEL.run(params=full(p), final_time=horizon,
                                            return_columns=["Contract Broken"]))
            rec[f"{rg}_slow_none"] = go("Slow", 0, 0)
            rec[f"{rg}_fast_none"] = go("Fast", 0, 0)
            rec[f"{rg}_current_both"] = go("Current", sig, D)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "montecarlo.csv", index=False)

    summary = {}
    for rg in REGIMES:
        s, f, b = df[f"{rg}_slow_none"], df[f"{rg}_fast_none"], df[f"{rg}_current_both"]
        both_or_neither = (np.isinf(s) == np.isinf(f))
        summary[rg] = {
            "F1 fast breaks no later than slow": float(np.mean(f <= s)),
            "F2 speed changes timing, not destination": float(np.mean(both_or_neither)),
            "F3 no policy: contract eventually breaks": float(np.mean(~np.isinf(s))),
            "F4 both policies: contract holds to 2225": float(np.mean(np.isinf(b))),
        }
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    labels = list(next(iter(summary.values())).keys())
    y = np.arange(len(labels))
    for k, (rg, col) in enumerate(zip(REGIMES, (C["blue"], C["orange"]))):
        vals = [summary[rg][l_] * 100 for l_ in labels]
        ax.barh(y + (k - 0.5) * 0.36, vals, height=0.34, color=col, label=rg)
        for yi, v in zip(y, vals):
            ax.text(v + 1, yi + (k - 0.5) * 0.36, f"{v:.0f} %", va="center", fontsize=8)
    ax.set_yticks(y, [l_[3:] for l_ in labels], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel(f"Share of {n} random parameter draws in which the finding holds")
    ax.set_title("Monte Carlo robustness of the main findings (horizon 2225)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2)
    save(fig, "S6_montecarlo")
    return summary


if __name__ == "__main__":
    tab, series = scenarios()
    fig_s1(series)
    fig_s2(tab)
    fig_s3(series)
    print("SCENARIOS (break year; bargaining power, legitimacy, concentration in 2075)")
    for _, r in tab.iterrows():
        print(f"  {r.regime:11s} {r.speed:7s} {r.policy:17s} {fmt_year(r.break_year):>6s}  "
              f"B={r.bargaining_2075:.2f} L={r.legitimacy_2075:.2f} P={r.concentration_2075:.2f}")
    cp = coup_proofing()
    print("COUP-PROOFING LOOP (no policy): break year with / without loop")
    for _, r in cp.iterrows():
        print(f"  {r.regime:11s} {r.speed:7s} {fmt_year(r.break_with_loop):>6s} / "
              f"{fmt_year(r.break_without_loop):>6s}   P2075 {r.concentration_2075_with:.2f} / "
              f"{r.concentration_2075_without:.2f}")
    tornado()
    if "--no-mc" not in sys.argv:
        summ = monte_carlo()
        print("MONTE CARLO (share of 1000 draws)")
        for rg, d in summ.items():
            for k, v in d.items():
                print(f"  {rg:11s} {k}: {v:.1%}")
    print(f"Figures: {FIG}  Results: {RES}")
