"""Benchmark gradient computation for steadystate solver on Kerr oscillator."""

import time
from typing import Any

import jax
import jax.numpy as jnp

import dynamiqs as dq

# Enable double precision
jax.config.update('jax_enable_x64', True)

TWOPI = 2 * jnp.pi


def build_kerr_oscillator(n: int, delta: float = 0.0):
    """Build a Kerr oscillator system."""
    a = dq.destroy(n)

    # Precomputed operator matrices
    a_jax = a.to_jax()
    adag_jax = a.dag().to_jax()
    adag2a2 = (a.dag() @ a.dag() @ a @ a).to_jax()
    adaga = (a.dag() @ a).to_jax()

    kap = 14.0 * TWOPI
    kerr = -1.0 * TWOPI
    ep = 16.0

    H = (
        -kerr / 2 * adag2a2
        - delta * adaga
        + 1j * jnp.sqrt(kap) * ep * a_jax
        - 1j * jnp.sqrt(kap) * ep * adag_jax
    )
    L = jnp.sqrt(kap) * a_jax
    return dq.asqarray(H), [dq.asqarray(L)]


def build_parametrized_kerr(n: int, delta: jax.Array):
    """Build a Kerr oscillator with delta as a differentiable parameter."""
    a = dq.destroy(n)

    # Precomputed operator matrices
    a_jax = a.to_jax()
    adag_jax = a.dag().to_jax()
    adag2a2 = (a.dag() @ a.dag() @ a @ a).to_jax()
    adaga = (a.dag() @ a).to_jax()

    kap = 14.0 * TWOPI
    kerr = -1.0 * TWOPI
    ep = 16.0

    H = (
        -kerr / 2 * adag2a2
        - delta * adaga
        + 1j * jnp.sqrt(kap) * ep * a_jax
        - 1j * jnp.sqrt(kap) * ep * adag_jax
    )
    L = jnp.sqrt(kap) * a_jax
    return dq.asqarray(H), [dq.asqarray(L)]


def expectation_value(n: int, delta: jax.Array, solver: Any):
    """Compute expectation value of a.dag() @ a."""
    H, Ls = build_parametrized_kerr(n, delta)
    result = dq.steadystate(H, Ls, solver=solver)
    a = dq.destroy(n)
    # Return <a.dag() a> = photon number
    return jnp.real(jnp.trace(result.rho.to_jax() @ (a.dag() @ a).to_jax()))


def benchmark_gradient(n: int = 40, n_runs: int = 5, warmup: int = 2):
    """Benchmark gradient computation for the steadystate solver."""
    print(f"\n{'='*60}")
    print(f"Benchmarking steadystate gradient for Kerr oscillator (n={n})")
    print(f"{'='*60}")

    solver = dq.SteadyStateGMRES(tol=1e-6, krylov_size=64, max_iteration=100)

    # Create the gradient function
    delta = jnp.array(0.0)

    # JIT compile the gradient function
    grad_fn = jax.jit(jax.grad(lambda d: expectation_value(n, d, solver)))

    # Warmup
    print('\nWarmup...')
    for i in range(warmup):
        grad = grad_fn(delta)
        grad.block_until_ready()
        print(f"  Warmup {i+1}/{warmup}: grad = {grad:.6f}")

    # Benchmark gradient
    print('\nBenchmarking gradient computation...')
    times_grad = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        grad = grad_fn(delta)
        grad.block_until_ready()
        t1 = time.perf_counter()
        times_grad.append(t1 - t0)
        print(f"  Run {i+1}/{n_runs}: {times_grad[-1]*1000:.2f} ms, grad = {grad:.6f}")

    avg_grad = sum(times_grad) / len(times_grad)
    print(f"\nAverage gradient time: {avg_grad*1000:.2f} ms")

    # Also benchmark forward pass for comparison
    print('\nBenchmarking forward pass (no gradient)...')
    fwd_fn = jax.jit(lambda d: expectation_value(n, d, solver))

    # Warmup forward
    for _ in range(warmup):
        val = fwd_fn(delta)
        val.block_until_ready()

    times_fwd = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        val = fwd_fn(delta)
        val.block_until_ready()
        t1 = time.perf_counter()
        times_fwd.append(t1 - t0)
        print(f"  Run {i+1}/{n_runs}: {times_fwd[-1]*1000:.2f} ms, val = {val:.6f}")

    avg_fwd = sum(times_fwd) / len(times_fwd)
    print(f"\nAverage forward time: {avg_fwd*1000:.2f} ms")
    print(f"Gradient / Forward ratio: {avg_grad/avg_fwd:.2f}x")

    return avg_fwd, avg_grad


if __name__ == '__main__':
    # Run benchmark
    fwd_time, grad_time = benchmark_gradient(n=40, n_runs=5, warmup=2)

    print(f"\n{'='*60}")
    print('Summary')
    print(f"{'='*60}")
    print(f"Forward pass:     {fwd_time*1000:.2f} ms")
    print(f"Gradient:         {grad_time*1000:.2f} ms")
    print(f"Ratio:            {grad_time/fwd_time:.2f}x")
