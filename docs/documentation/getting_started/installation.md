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

## GPU support

Dynamiqs leverages [JAX](https://jax.readthedocs.io/) for high-performance computing. By default, JAX is installed with CPU-only support. To enable **GPU acceleration**, you need to install a GPU-compatible version of JAX. Please refer to the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for detailed, platform-specific instructions on installing JAX with CUDA (NVIDIA) or other accelerator support.

!!! tip
    Install the GPU-enabled version of JAX **before** installing Dynamiqs to avoid dependency conflicts.


## Reinstalling dynamiqs

You may occasionally need to reinstall Dynamiqs, for instance when upgrading to a newer release. Although modern 
package managers such as [`uv`](https://docs.astral.sh/uv/)￼ handle upgrades and dependency resolution automatically, 
this is not always true when using plain pip. In that case, uninstalling Dynamiqs alone can leave residual dependencies 
from the previous installation, potentially causing version conflicts. To guarantee a clean reinstallation, 
we recommend uninstalling Dynamiqs together with its main JAX dependencies:

```bash
pip uninstall jax jaxlib optax diffrax jaxtyping equinox dynamiqs
pip install dynamiqs
```
