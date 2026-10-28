# Installation

## Install with pip

Dynamiqs requires **Python 3.11 or later**. The recommended way to install Dynamiqs is with `pip`, or with a package manager such as [`uv`](https://docs.astral.sh/uv/) or [`pixi`](https://pixi.prefix.dev/latest/). To install Dynamiqs with `pip`, simply run:

```shell
pip install dynamiqs
```

## Install from source

To install the latest development version directly from GitHub:

```shell
pip install git+https://github.com/dynamiqs/dynamiqs.git
```

To set up a development environment (editable install, tests and documentation tooling), see the [contributing guide](../../community/contributing.md).

## GPU support

Dynamiqs leverages [JAX](https://docs.jax.dev/) for high-performance computing. By default, JAX is installed with CPU-only support. To enable **GPU acceleration** on NVIDIA GPUs, install Dynamiqs together with the CUDA-enabled version of JAX:

```shell
pip install -U dynamiqs "jax[cuda13]"
```

Use `"jax[cuda12]"` instead if your NVIDIA driver does not support CUDA 13. These wheels are only available on **Linux**: on Windows, use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/) (where NVIDIA GPU support in JAX is experimental). Please refer to the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for the supported driver versions and for other accelerators.

To check that JAX detects your GPU, run:

```shell
python -c "import jax; print(jax.devices())"
```

which should list a `CudaDevice`. Dynamiqs then runs on the GPU by default. To select a device explicitly, use [`dq.set_device()`][dynamiqs.set_device], and see the [Devices and precision](../basics/devices-and-precision.md) tutorial for more details.

## Reinstalling Dynamiqs

You may occasionally need to reinstall Dynamiqs, for instance when upgrading to a newer release. Although modern package managers such as [`uv`](https://docs.astral.sh/uv/) handle upgrades and dependency resolution automatically, this is not always true when using plain `pip`. In that case, uninstalling Dynamiqs alone can leave residual dependencies from the previous installation, potentially causing version conflicts. To guarantee a clean reinstallation, we recommend uninstalling Dynamiqs together with its main JAX dependencies (and the JAX CUDA plugins, if you installed GPU support):

```shell
pip uninstall jax jaxlib diffrax equinox jaxtyping optimistix dynamiqs
pip uninstall jax-cuda13-plugin jax-cuda13-pjrt jax-cuda12-plugin jax-cuda12-pjrt
pip install dynamiqs
```
