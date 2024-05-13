# Deep Equilibrium Nets for The Analytic Climate Economy

This GitHub repository contains the codes for my master's thesis, called "Beyond the Curse of Dimensionality? Deep Equilibrium Nets for the Analytic Climate Economy". Please refer to Azinovib et al. (2022) for Deep Equilibrium Nets and their [![GitHub Repository](https://img.shields.io/badge/GitHub-DeepEquilibriumNets-blue?logo=github)](https://github.com/sischei/DeepEquilibriumNets) and to Traeger (2023) for the paper on the Analytic Climate Economy.

## Dependencies

- debugpy v:1.6.6
- Jinja2 v:3.1.3
- Keras v:2.10.0
- Keras-Preprocessing
- matplotlib v:3.8.3
- matplotlib-inline v:0.1.6
- numpy v:1.26.3
- pandas v:2.2.2
- pytest v:8.0.0
- PyYAML v:6.0.1
- scipy v:1.11.4
- tensorboard v:2.10.0
- tensorflow v:2.10.0
- tensorflow-estimator v:2.10.0
- tensorflow-probability v:0.24.0

## Usage

Model training is done by running

```shell
python run_ace_dice.py --model_version version
```

where version is replaced by either 2016 or 2023. This loads the most recently trained weights, located in [![Logs Directory](https://img.shields.io/badge/Logs-Directory-lightgrey?logo=github)](logs/). If you want to train a network from scratch, you need to delete the checkpoint files before starting training.

For storing the results with the training weights, run

```shell
python generate_results.py --model_version version
```

where you also need to replace version with 2016 or 2023.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

Azinovic, M., Gaegauf, L., & Scheidegger, S. (2022). Deep Equilibrium Nets. International Economic Review, 63(4), 1471–1525.

Traeger, C. P. (2023). ACE—Analytic Climate Economy. American Economic Journal. Economic Policy, 15(3), 372–406.
