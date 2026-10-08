# VQLS + TREX/ZNE

Variational Quantum Linear Solver with noise mitigation (TREX, ZNE) for four PDE families: Heat1D, Heat2D, Beam1D, Beam2D.

## Installation

git clone <repo-url> vqls_full
cd vqls_full
python -m venv .venv && source .venv/bin/activate
pip install -e .

## Data

This repo contains code only. Data lives outside the repo.

By default, config.py looks in $HOME:

    $HOME/Heat1D/                        K matrices, solutions
    $HOME/Heat2D/
    $HOME/beam1d_final_results/
    $HOME/Beam 2D/
    $HOME/other_families_trex_campaign/  training campaigns
    $HOME/beam1d_optimized/
    $HOME/ideal_preconditioning_32dof/   preconditioner study
    $HOME/ibm_fixed_theta_vqls/          hardware results

Override with:

    export VQLS_DATA=/path/to/data

## Usage

    python scripts/make_figs.py --fig all

## License

MIT
