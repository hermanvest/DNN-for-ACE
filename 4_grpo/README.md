# 4. Extension: GRPO for ACE

Group Relative Policy Optimization (GRPO) applied to the Analytic Climate Economy, as a
model-free counterpart to the model-based actor-critic in [`../3_actor_critic/`](../3_actor_critic/).
This directory is self-contained: it has its own copy of the environment and benchmark, its
own training code, results and tests. It does not import from parts 1–3.

## Idea

GRPO is PPO without a critic. For each prompt it samples a group of responses and uses each
response's reward, normalized within the group, as its advantage. Applied to ACE:

- **Policy:** a Gaussian in logit space over the consumption rate `x` and the abatement rate
  `μ`, with a learnable standard deviation. Evaluation uses the mean action.
- **Prompts and groups:** a prompt is a start state on the current policy's trajectory. From
  each, a group of 16 trajectories of 40 periods is sampled.
- **Advantages:** *process* (default): each action's discounted return-to-go, normalized across
  the group at the same step. *Outcome* (ablation): the trajectory's discounted return,
  normalized across the group and shared by all of its actions.
- **Update:** the PPO clipped objective, 4 epochs of 4 mini-batches per iteration. There is no
  KL penalty, since there is no reference policy.
- **Model-free:** the environment is only sampled. Unlike the actor-critic, nothing is
  differentiated through the dynamics, and there is no value function.

Without a critic there are no costates, so the SCC is read off the policy instead: the marginal
abatement cost at the chosen abatement rate, which equals the SCC at an optimum wherever full
abatement does not bind.

## Layout

```
4_grpo/
├── code/
│   ├── ace_grpo/
│   │   ├── env.py         Differentiable ACE-DICE environment (copy of part 3)
│   │   ├── benchmark.py   Closed-form costates and optimal abatement (copy of part 3)
│   │   ├── exogenous.py   DICE 2016 / 2023 exogenous paths (copy of part 3)
│   │   └── grpo.py        GRPO
│   ├── configs/           Model and GRPO settings
│   ├── tests/             The benchmark is consistent with the dynamics
│   ├── train.py           python train.py --version 2016
│   └── make_results.py    Figures, tables and CSVs
└── results/               Figures, tables, trajectories, weights and logs
```

## Reproducing

```bash
cd 4_grpo/code
python -m pytest
python train.py --version 2016
python train.py --version 2023
python train.py --version 2016 --advantage outcome --tag _outcome   # ablation
python make_results.py
```

`make_results.py` also reads the saved results of the DEQN (`../2_revised/results/`) and the
actor-critic (`../3_actor_critic/results/`), as data only, for the comparison table and the
learning-curve figure.
