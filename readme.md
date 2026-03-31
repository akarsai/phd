# Code for PhD Thesis *"Nonlinear Energy-Based Systems: Modeling, Control and Numerical Realization"*

This repository contains the code for the numerical experiments in the thesis.

## Reproducing the results

1. Install [`uv`](https://github.com/astral-sh/uv) by following the [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
2. Copy the code and switch folder. Using `git`, the commands are:
```shell
git clone https://github.com/akarsai/phd.git && cd phd
``` 
3. Run experiments (the plots will be placed in the `results/figures` directory):
```shell
uv run python plots/dg_intro.py
```
```shell
uv run python plots/dg_qsr.py
```
```shell
uv run python plots/dg.py
```
```shell
uv run python plots/mpg_cahn_hilliard_state.py
```
```shell
uv run python plots/mpg_convergence.py 
```
```shell
uv run python plots/mpg_doubly_nonlinear_parabolic_state.py
```
```shell
uv run python plots/mpg_energybalance.py
```
```shell
uv run python plots/oc_casadi.py
```
```shell
uv run python plots/oc_state_adjoint.py
```
```shell
uv run python plots/pc_passivity.py
```
```shell
uv run python plots/pc_performance.py
```



## Some hints
- Throughout the codebase, the time index is always at position `0`. The state `z` thus is stored in an array with shape `z.shape == (number_of_timepoints, dimension)`.
- Since the implementation uses the algorithmic differentiation capabilities of JAX, the implementations of all functions need to be written in a JAX-compatible fashion. The provided examples should be a good starting point.
- In case of questions, feel free to reach out.