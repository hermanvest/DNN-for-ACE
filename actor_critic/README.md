# Model-based actor-critic for ACE

A reinforcement-learning alternative to Deep Equilibrium Nets for the Analytic Climate
Economy. This directory is independent of the thesis (`../revised/`) and of the original
code in the repository root: it has its own environment, benchmark, training code, results
and paper.

**Paper:** [`paper/paper.pdf`](paper/paper.pdf)

## Idea

A DEQN trains a network to satisfy hand-derived equilibrium conditions. If a condition is
derived wrongly, the network converges to the wrong solution. This directory asks whether
learning from the reward alone (log consumption), with no hand-derived conditions, can match
it.

- **Critic** `V(s)`: trained on the Bellman equation of the current policy, and on its
  gradient. The target gradient `∇ₛ[u + βV(f(s, π(s)))]` is computed by automatic
  differentiation through the known dynamics (value-gradient learning).
- **Actor** `π(s) = (x, μ)`: maximizes `u(s, π(s)) + βV(f(s, π(s)))`, differentiating through
  the dynamics.
- **Costates and the SCC** are read off the critic afterwards as `β∇V(s')`. No network
  outputs them.

Business-as-usual emissions follow an exogenous reference path (the DICE definition evaluated
along the analytic optimal path), so that the closed-form ACE solution is exact for the
problem the actor-critic solves.

## Layout

```
actor_critic/
├── code/
│   ├── ace_ac/
│   │   ├── env.py         Differentiable ACE-DICE environment and analytic policy
│   │   ├── benchmark.py   Closed-form costates and optimal abatement
│   │   ├── exogenous.py   DICE 2016 / 2023 exogenous paths
│   │   └── agent.py       Model-based actor-critic with value-gradient learning
│   ├── configs/           Model and training settings
│   ├── tests/             The benchmark is consistent with the dynamics
│   ├── train.py           python train.py --version 2016
│   └── make_results.py    Figures, tables and CSVs
├── results/               Figures, tables, trajectories, weights and logs
└── paper/                 LaTeX source and compiled PDF
```

## Reproducing

```bash
cd actor_critic/code
python -m pytest
python train.py --version 2016
python train.py --version 2023
python train.py --version 2016 --gradient-weight 0 --tag _td_only        # ablation
python train.py --version 2016 --copies 0 --tag _no_exploration          # ablation
python make_results.py
cd ../paper && latexmk -pdf main.tex && cp main.pdf paper.pdf
```

`make_results.py` also reads the trajectories of the corrected DEQN from
`../revised/results/csv/`, but only for the comparison table. If they are missing, the DEQN
columns are left empty.
