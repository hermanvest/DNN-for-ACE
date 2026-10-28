# Deep Equilibrium Nets for The Analytic Climate Economy

> [!IMPORTANT]
> **Corrections and an updated thesis are available.** After submission, errors were found in the
> equilibrium conditions (loss functions) of the thesis and of the original code: the capital
> envelope condition, double-counted terms in the consumption and emissions first-order
> conditions, a missing unit conversion, off-by-one time indices, and a double transformation of
> emissions in the Bellman residual. They are corrected in a rewritten implementation, and the
> updated thesis, with new results, is in [`revised/`](revised/) ([PDF](revised/thesis/thesis_revised.pdf)). The original code used for the
> submitted thesis is kept unchanged in the repository root.

This GitHub repository contains the codes for my master's thesis, called "Beyond the Curse of Dimensionality? Deep Equilibrium Nets for the Analytic Climate Economy". Please refer to Azinovic et al. (2022) for Deep Equilibrium Nets and their [![GitHub Repository](https://img.shields.io/badge/GitHub-DeepEquilibriumNets-blue?logo=github)](https://github.com/sischei/DeepEquilibriumNets) and to Traeger (2023) for the paper on the Analytic Climate Economy.

## Repository layout

The repository is divided into three parts:

| Part | Location | Contents |
|---|---|---|
| **Revised version** | [`revised/`](revised/) | Corrected and rewritten code ([`revised/code/`](revised/code/)), new results with figures and tables ([`revised/results/`](revised/results/)), and the updated thesis ([PDF](revised/thesis/thesis_revised.pdf), LaTeX source in [`revised/thesis/`](revised/thesis/)). See [`revised/README.md`](revised/README.md) for what was corrected and how to reproduce the results. |
| **Actor-critic follow-up** | [`actor_critic/`](actor_critic/) | A separate research project: a model-based actor-critic that learns ACE from the reward alone, without hand-derived equilibrium conditions, with its own code, results and paper ([PDF](actor_critic/paper/paper.pdf)). See [`actor_critic/README.md`](actor_critic/README.md). |
| **Original thesis code** | repository root (`agents/`, `configs/`, `environments/`, `networks/`, `training_algorithms/`, `plotting/`, `logs/`, ...) | The code, trained checkpoints and results used for the submitted thesis, kept unchanged for reference. Note that its loss functions contain the errors described above. |

## Revised version

```shell
cd revised/code
python -m pytest                    # the analytic solution satisfies the corrected conditions
python train.py --version 2016      # or 2023
python make_results.py              # figures, tables and CSVs in revised/results/
```

## Original thesis code

### Usage

The main results for the thesis, including the analytical calculations, are in the following notebook: [![Python Notebook](https://img.shields.io/badge/Python%20Notebook-thesis__results.ipynb-lightgrey?logo=jupyter)](plotting/thesis_results.ipynb)

Model training is done by running

```shell
python run_ace_dice.py --model_version version
```

where version is replaced by either 2016 or 2023. This loads the most recently trained weights, located in [![Logs Directory](https://img.shields.io/badge/Logs-Directory-lightgrey?logo=github)](logs/). If you want to train a network from scratch, you need to delete the checkpoint files before starting training.

For storing the results with the training weights, run

```shell
python generate_results.py --model_version version
```

where version is replaced by 2016 or 2023.


Training logs can be analyzed with [![TensorBoard](https://img.shields.io/badge/TensorBoard-Open%20docs-orange?logo=tensorflow)](https://www.tensorflow.org/tensorboard)
. To run tensorboard and analyze logged training results, use

```shell
tensorboard --logdir=logs/version/training_stats
```

where version is replaced by 2016 or 2023.

### Dependencies

- ![debugpy](https://img.shields.io/badge/debugpy-v1.6.6-blue)
- ![Jinja2](https://img.shields.io/badge/Jinja2-v3.1.3-blue)
- ![Keras](https://img.shields.io/badge/Keras-v2.10.0-blue)
- ![matplotlib](https://img.shields.io/badge/matplotlib-v3.8.3-blue)
- ![matplotlib-inline](https://img.shields.io/badge/matplotlib--inline-v0.1.6-blue)
- ![numpy](https://img.shields.io/badge/numpy-v1.26.3-blue)
- ![pandas](https://img.shields.io/badge/pandas-v2.2.2-blue)
- ![pytest](https://img.shields.io/badge/pytest-v8.0.0-blue)
- ![PyYAML](https://img.shields.io/badge/PyYAML-v6.0.1-blue)
- ![scipy](https://img.shields.io/badge/scipy-v1.11.4-blue)
- ![tensorboard](https://img.shields.io/badge/tensorboard-v2.10.0-blue)
- ![tensorflow](https://img.shields.io/badge/tensorflow-v2.10.0-blue)
- ![tensorflow-estimator](https://img.shields.io/badge/tensorflow--estimator-v2.10.0-blue)
- ![tensorflow-probability](https://img.shields.io/badge/tensorflow--probability-v0.24.0-blue)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

Azinovic, M., Gaegauf, L., & Scheidegger, S. (2022). Deep Equilibrium Nets. International Economic Review, 63(4), 1471–1525.

Traeger, C. P. (2023). ACE—Analytic Climate Economy. American Economic Journal. Economic Policy, 15(3), 372–406.
