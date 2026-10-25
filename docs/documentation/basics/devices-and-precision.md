# Devices and precision

Dynamiqs runs on CPUs, GPUs and TPUs through [JAX](https://docs.jax.dev/). This tutorial explains how the device is chosen, how to change it, and how to set the floating-point precision of simulations. These choices are made through a few global settings, which are summarized at the end of this page.

```python
import dynamiqs as dq
import jax
```

## Choosing a device

### How JAX picks a device

When it starts, JAX looks for the available backends and uses the most capable one by default: a TPU if there is one, otherwise a GPU, otherwise the CPU. You do not need to change your code to run on a GPU: if JAX detects one, all qarrays are created on it and all simulations run on it. You can list the devices JAX sees with `jax.devices()`, and check the default backend with `jax.default_backend()`:

<!-- skip: start -->
```pycon
>>> jax.devices()
[CudaDevice(id=0)]
>>> jax.default_backend()
'gpu'
```
<!-- skip: end -->

By default, JAX is installed with CPU support only, in which case `jax.devices()` only lists a `CpuDevice`. To run on an NVIDIA GPU, you must install the CUDA-enabled version of JAX, see the [Installation](../getting_started/installation.md#gpu-support) page.

The `.devices()` method returns the devices on which a qarray is stored:

<!-- skip: start -->
```pycon
>>> dq.sigmax().devices()
{CudaDevice(id=0)}
```
<!-- skip: end -->

### Selecting a device

To run on another device than the default one, use [`dq.set_device()`][dynamiqs.set_device], which takes the kind of device (`'cpu'`, `'gpu'` or `'tpu'`) and optionally its index:

```python
dq.set_device('cpu')
```

<!-- restore the default device
```python
jax.config.update('jax_default_device', None)
```
-->

Qarrays created afterwards are stored on this device, and simulations whose inputs are on this device run on it. Requesting a device that JAX cannot find raises an error, for example `dq.set_device('gpu')` on a machine without a GPU, or with the CPU-only version of JAX.

[`dq.set_device()`][dynamiqs.set_device] is a shortcut for setting the `jax_default_device` option of JAX, so the usual JAX tools also work with qarrays: the `jax.default_device()` context manager selects a device for a block of code only, and `jax.device_put()` moves existing qarrays to a device:

```python
cpu = jax.devices('cpu')[0]

with jax.default_device(cpu):
    a = dq.destroy(8)  # created on the CPU

H = jax.device_put(dq.sigmax(), cpu)  # moved to the CPU
```

See the [JAX documentation on data placement](https://docs.jax.dev/en/latest/faq.html#faq-data-placement) for more details.

!!! Tip "Hiding the GPU from JAX"
    To force JAX to use the CPU even when a GPU is available, for example to compare run times, set the `JAX_PLATFORMS` environment variable before starting Python:

    ```shell
    JAX_PLATFORMS=cpu python simulation.py
    ```

## Floating-point precision

### Single and double precision

By default, Dynamiqs uses **single precision**: real numbers are stored as `float32` and complex numbers as `complex64`, which gives a relative accuracy of about $10^{-7}$. This matches the default of JAX, and is enough for most simulations, but it differs from NumPy and QuTiP, see [the sharp bits](../getting_started/sharp-bits.md#floating-point-precision) for its limitations.

To switch to **double precision** (`float64` and `complex128`, about $10^{-16}$), use [`dq.set_precision()`][dynamiqs.set_precision]:

```pycon
>>> dq.sigmax().dtype
dtype('complex64')
>>> dq.set_precision('double')
>>> dq.sigmax().dtype
dtype('complex128')
>>> dq.set_precision('single')  # back to the default precision
```

Double precision is useful when you need tight solver tolerances (`rtol` and `atol` well below $10^{-6}$), when quantities of very different magnitudes are involved, or to check that a single-precision result has converged.

It has a cost: every array takes twice as much memory, and each operation is slower. On CPUs and on data-center GPUs (for example NVIDIA V100, A100 or H100), double precision is typically up to about twice as slow as single precision. On **consumer GPUs**, such as most NVIDIA GeForce cards, double-precision arithmetic is only a small fraction of the single-precision throughput (1/32 or 1/64 on recent generations), and simulations can be an order of magnitude slower or more.

!!! Warning "Set the precision before creating any array"
    [`dq.set_precision()`][dynamiqs.set_precision] only applies to arrays created after it is called: existing qarrays keep their data type, and mixing arrays created before and after the call either computes with single-precision inputs or raises an error. Call it once, at the top of your script or notebook, right after importing Dynamiqs:

    ```python
    import dynamiqs as dq

    dq.set_precision('double')
    ```

    This is a shortcut for the `jax_enable_x64` option of JAX, which can also be set with the `JAX_ENABLE_X64=1` environment variable.

<!-- set precision back to default
```python
dq.set_precision('single')
```
-->

### Precision of matrix multiplications

On GPUs and TPUs, the hardware can carry out a single-precision matrix multiplication with fewer significant bits internally, for example in `bfloat16` on TPUs or in TensorFloat-32 on recent NVIDIA GPUs. This is much faster, but the products are then only accurate to about three decimal digits, and JAX allows it by default on some of these devices. Quantum simulations chain many matrix products, in which such errors accumulate, for example into a drift of the norm of the state. **Dynamiqs therefore sets the matrix multiplication precision to `'highest'` upon import**, which keeps full single precision for every product on every device.

You can trade this accuracy for speed with [`dq.set_matmul_precision()`][dynamiqs.set_matmul_precision]:

```python
dq.set_matmul_precision('high')  # 'highest' by default
```

<!-- restore the default matmul precision
```python
dq.set_matmul_precision('highest')
```
-->

This setting only affects single-precision matrices on GPUs and TPUs: it has no effect on CPUs, or in double precision. Adaptive methods compensate for the lost accuracy by taking more steps, so the gain is not guaranteed: see the [Performance](performance.md#trade-accuracy-for-speed) tutorial before relying on it.

## Global settings

Dynamiqs has six global settings, which all apply to the rest of the program:

| Function | Default | Effect |
| --- | --- | --- |
| [`dq.set_device()`][dynamiqs.set_device] | chosen by JAX | Device on which qarrays are created and simulations run. |
| [`dq.set_precision()`][dynamiqs.set_precision] | `'single'` | Floating-point precision of new arrays. |
| [`dq.set_matmul_precision()`][dynamiqs.set_matmul_precision] | `'highest'` | Internal precision of matrix multiplications on GPUs and TPUs. |
| [`dq.set_layout()`][dynamiqs.set_layout] | `'dia'` | Default layout of operators, see [Quantum arrays and layouts](qarrays-and-layouts.md#dense-and-sparse-diagonal-layouts). |
| [`dq.set_composite()`][dynamiqs.set_composite] | `False` | Whether operators of several subsystems are stored as their factors, see [Composite qarrays](composite-qarrays.md). |
| [`dq.set_progress_meter()`][dynamiqs.set_progress_meter] | `True` | Progress meter displayed by solvers. |

The last three do not depend on the hardware. [`dq.set_layout()`][dynamiqs.set_layout] chooses whether operators such as [`dq.destroy()`][dynamiqs.destroy] are stored in the sparse diagonal layout or as dense arrays, and [`dq.set_composite()`][dynamiqs.set_composite] whether `dq.destroy(*dims)` and [`dq.tensor()`][dynamiqs.tensor] store the operators of several subsystems as their factors. [`dq.set_progress_meter()`][dynamiqs.set_progress_meter] takes `True` for a [tqdm](https://github.com/tqdm/tqdm) progress bar, `False` to disable it, or a progress meter such as [`dq.TextProgressMeter()`][dynamiqs.TextProgressMeter], which prints plain text instead:

```python
dq.set_progress_meter(dq.TextProgressMeter())
```

<!-- restore the default progress meter
```python
dq.set_progress_meter(True)
```
-->

Each solver also takes a `progress_meter` argument, which overrides the global setting for one call. The progress meter has a small cost at every solver step, so it is worth disabling for short or repeated simulations, see the [Performance](performance.md#turn-off-the-progress-meter) tutorial.
