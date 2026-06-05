"""Main logic to run during benchmark processes."""

from cross_resonance_modulated_sesolve import build_problem

import dynamiqs as dq


def main() -> None:
    hamiltonian, initial_state, tsaves = build_problem()

    result = dq.sesolve(hamiltonian, initial_state, tsaves, progress_meter=False)
    result.block_until_ready()

    # NOTE: this is just for testing for now
    print(result)


if __name__ == '__main__':
    main()


__all__ = ['main']
