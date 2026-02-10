# Installation

You can install Dynamiqs with `pip`:

```shell
pip install dynamiqs
```

!!! Note
    If you're using a GPU, please refer to the [JAX installation](https://jax.readthedocs.io/en/latest/installation.html) documentation page for detailed instructions on how to install JAX for your device.

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
