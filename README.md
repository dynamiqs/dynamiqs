![torchqdynamics logo](./media/torchqdynamics_logo.png)
Quantum systems simulation with PyTorch.

This library provides differentiable solvers for the Schrödinger Equation, the Lindblad Master Equation and the Stochastic Master Equation. All the solvers are implemented using PyTorch and can run on GPUs.

:hammer_and_wrench: This library is under construction and while the APIs and solvers are still finding their footing, we're working hard to make it worth the wait. Check back soon for the grand opening!

## Installation
Clone the repository, install the dependencies and install the repository in editable mode in any Python virtual environment:
```shell
# pip
pip install -e .

# conda
conda develop .
```

## Usage
:construction: WIP

## Performance
:construction: WIP

## Contribute
Install developer dependencies:
```shell
pip install -e ".[dev]"
```

Run the following before each commit:
```shell
isort .  # sort the imports
black .  # auto-format the code
pytest  # run the test suite
```

Alternatively you can use `pre-commit` to run automatically the sorting (isort)
and auto-formatting (black) automatically before each commit:
```shell
pip install pre-commit
pre-commit install
```
