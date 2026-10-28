# Deep Equilibrium Nets for The Analytic Climate Economy

> [!IMPORTANT]
> **Corrections and an updated thesis are available.** After submission, errors were found in the
> equilibrium conditions (loss functions) of the thesis and of the original code: the capital
> envelope condition, double-counted terms in the consumption and emissions first-order
> conditions, a missing unit conversion, off-by-one time indices, and a double transformation of
> emissions in the Bellman residual. They are corrected in a rewritten implementation, and the
> updated thesis, with new results, is in [`2_revised/`](2_revised/)
> ([PDF](2_revised/thesis/thesis_revised.pdf)). The original code used for the submitted thesis
> is kept unchanged in [`1_original/`](1_original/).

This GitHub repository contains the codes for my master's thesis, called "Beyond the Curse of Dimensionality? Deep Equilibrium Nets for the Analytic Climate Economy". Please refer to Azinovic et al. (2022) for Deep Equilibrium Nets and their [![GitHub Repository](https://img.shields.io/badge/GitHub-DeepEquilibriumNets-blue?logo=github)](https://github.com/sischei/DeepEquilibriumNets) and to Traeger (2023) for the paper on the Analytic Climate Economy.

## Repository layout

The repository is divided into separate, numbered parts. Each part is self-contained, with its
own code, results and README, and later extensions get the next number.

| | Part | Contents |
|---|---|---|
| 1 | [`1_original/`](1_original/) | The code, trained checkpoints and results used for the submitted thesis, unchanged. Its loss functions contain the errors described above. |
| 2 | [`2_revised/`](2_revised/) | The revision: corrected and rewritten code, new results, and the updated thesis ([PDF](2_revised/thesis/thesis_revised.pdf)). |
| 3 | [`3_actor_critic/`](3_actor_critic/) | An extension: a model-based actor-critic that learns ACE from the reward alone, without hand-derived equilibrium conditions, with its own paper ([PDF](3_actor_critic/paper/paper.pdf)). |

See the README in each directory for how to run it.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

Azinovic, M., Gaegauf, L., & Scheidegger, S. (2022). Deep Equilibrium Nets. International Economic Review, 63(4), 1471–1525.

Traeger, C. P. (2023). ACE—Analytic Climate Economy. American Economic Journal. Economic Policy, 15(3), 372–406.
