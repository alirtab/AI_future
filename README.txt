AI Future: Reproducibility Package
==================================

This repository contains the model, scripts, and generated result tables for:

When the State No Longer Needs the Nation: AI and the Future of the Social Contract

The repository is associated with the manuscript at:
https://github.com/alirtab/AI_future

Contents
--------

causal_loops.py       Generate the causal-loop diagram (Figure F1).
game_theory.py        Solve Games A and B and generate their figures.
sd_runs.py             Run the system-dynamics scenarios and robustness checks.
nation_state.mdl       Vensim model used by the system-dynamics analysis.
requirements.txt       Python package versions used for the analyses.
results/               Scenario, sensitivity, and Monte Carlo output tables.
figures/               Generated figures (created by the scripts).

Requirements
------------

Python 3.10 or newer is recommended. Install the pinned dependencies with:

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt

Reproduction
------------

Run the scripts from the repository root:

    .venv/bin/python game_theory.py
    .venv/bin/python causal_loops.py
    .venv/bin/python sd_runs.py --no-mc
    .venv/bin/python sd_runs.py

The first system-dynamics command skips the 1,000-draw Monte Carlo analysis;
the final command runs the complete analysis and may take several minutes.
Figures are written to figures/ and tabular outputs to results/.

The model parameters are illustrative and normalized. The outputs are
conditional scenarios, not forecasts.

AI-assisted editing
--------------------

AI tools were used only to improve English wording and proofreading. The
authors checked the manuscript, code, model, and results and take responsibility
for the final contents.

License and citation
--------------------

No software license has been specified yet. Cite the associated paper when
using this package.
