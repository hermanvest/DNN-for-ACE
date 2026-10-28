# 1. Original thesis code

The code, trained checkpoints and results used for the submitted master's thesis, *Beyond the
Curse of Dimensionality? Deep Equilibrium Nets for the Analytic Climate Economy*. It is kept
unchanged for reference.

> [!WARNING]
> The loss functions in this code contain errors in the equilibrium conditions, so the results
> here do not match the analytic solution. The corrected code, results and thesis are in
> [`../2_revised/`](../2_revised/).

All commands below are run from this directory (`1_original/`).

## Usage

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

## Dependencies

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
