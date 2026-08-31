"""CLI entry point: `python -m benchmarks [run options]` or `compare A B`."""

from __future__ import annotations

import argparse
import sys

from .cases import Tier
from .compare import compare
from .runner import run_suite


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv

    if argv[:1] == ['compare']:
        parser = argparse.ArgumentParser(
            prog='python -m benchmarks compare',
            description='Compare two benchmark result files.',
        )
        parser.add_argument('before', help='baseline results JSON file')
        parser.add_argument('after', help='new results JSON file')
        args = parser.parse_args(argv[1:])
        compare(args.before, args.after)
    else:
        parser = argparse.ArgumentParser(
            prog='python -m benchmarks',
            description='Run the dynamiqs timing benchmarks.',
            epilog='To compare two result files: python -m benchmarks compare A B',
        )
        parser.add_argument(
            '--filter', help='only run cases whose key contains this substring'
        )
        parser.add_argument(
            '--tier',
            type=Tier,
            choices=list(Tier),
            help='only run the cases of this tier (default: all tiers)',
        )
        parser.add_argument(
            '--quick', action='store_true', help='use tiny problem sizes'
        )
        parser.add_argument(
            '--repeats', type=int, default=5, help='timed runs per case (default: 5)'
        )
        parser.add_argument('--out', help='write results to this JSON file')
        args = parser.parse_args(argv)
        run_suite(
            quick=args.quick,
            filter_=args.filter,
            repeats=args.repeats,
            out=args.out,
            tier=args.tier,
        )


if __name__ == '__main__':
    main()
