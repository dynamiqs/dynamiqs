# ruff: noqa: T201  (command-line script reporting its progress)
"""Rebuild the documentation of past releases with the current theme, and deploy each
release as its own version on the local `gh-pages` branch with mike.

For each tag, the release source is checked out in a temporary git worktree, its
doctests are run in a virtual environment resolved as of the release date (they
generate the figures), its Python API pages are generated with its own
`generate_python_api.py`, and the current theme and configuration are copied over it,
keeping the release navigation.

Run from the repository root, with the Zensical fork of mike installed:

    uv pip install "mike @ git+https://github.com/squidfunk/mike.git"
    uv run python docs/backfill_versions.py            # all tags from v0.2.0
    uv run python docs/backfill_versions.py v0.3.7     # selected tags

Nothing is pushed: review the result with `mike serve`, then push `gh-pages`.
"""

import datetime
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tomllib
from pathlib import Path

import tomli_w
import yaml
from griffe_strip_lines import pattern

ROOT = Path(__file__).resolve().parent.parent
FIRST_TAG = (0, 2, 0)
THEME_FILES = [
    'docs/overrides',
    'docs/stylesheets',
    'docs/javascripts',
    'docs/templates',
    'docs/griffe_strip_lines.py',
]

# stands in for `mkdocs_gen_files` in the `generate_python_api.py` of past releases
GEN_FILES_SHIM = """\
from pathlib import Path


def open(path, mode='r', **kwargs):
    path = Path('docs', path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open(mode, **kwargs)


def set_edit_path(*args, **kwargs):
    pass
"""


# a mkdocstrings directive followed by its indented YAML options
DIRECTIVE = re.compile(r'^(::: \S+\n)((?:    .*\n?)+)', flags=re.MULTILINE)


class _IgnorePythonTagsLoader(yaml.SafeLoader):
    pass


# `mkdocs.yml` uses `!!python/name:` tags for the emoji extension, which only matter
# to MkDocs
_IgnorePythonTagsLoader.add_multi_constructor(
    'tag:yaml.org,2002:python/', lambda *_: None
)


def run(*args, **kwargs):
    print('$', *args, file=sys.stderr, flush=True)
    return subprocess.run(args, check=True, **kwargs)


def release_tags():
    tags = run('git', 'tag', '--list', 'v*', capture_output=True, text=True, cwd=ROOT)
    versions = [tuple(map(int, tag[1:].split('.'))) for tag in tags.stdout.split()]
    return [f'v{".".join(map(str, v))}' for v in sorted(versions) if v >= FIRST_TAG]


def run_doctests(worktree, tag):
    """Run the release doctests to generate its figures, return whether they passed."""
    tag_date = run(
        'git', 'log', '-1', '--format=%cs', tag, capture_output=True, text=True
    ).stdout.strip()
    # resolve the dependencies as they were on the release day, later releases of the
    # dependencies may have broken the release doctests
    exclude_newer = datetime.date.fromisoformat(tag_date) + datetime.timedelta(days=1)
    venv_python = worktree / '.venv' / 'bin' / 'python'
    run('uv', 'venv', '--quiet', '--python', '3.11', worktree / '.venv')
    run(
        'uv', 'pip', 'install', '--quiet', '--python', venv_python,
        '--exclude-newer', exclude_newer.isoformat(), '-e', f'{worktree}[dev]',
    )  # fmt: skip
    for figs_dir in ('docs/figs_code', 'docs/figs_docs'):
        (worktree / figs_dir).mkdir(parents=True, exist_ok=True)
    pytest = [venv_python, '-m', 'pytest', '-q', '-p', 'no:cacheprovider']
    doctests = subprocess.run([*pytest, 'dynamiqs', 'docs'], cwd=worktree, check=False)
    return doctests.returncode == 0


def move_custom_options_to_extra(markdown):
    """Move the `namespace` and `table` options of the mkdocstrings directives under
    `extra`, the only place where current mkdocstrings-python accepts custom options.
    """

    def move(match):
        options = yaml.safe_load(textwrap.dedent(match.group(2))).get('options', {})
        custom = {
            key: options.pop(key) for key in ('namespace', 'table') if key in options
        }
        if not custom:
            return match.group(0)
        options.setdefault('extra', {}).update(custom)
        dumped = yaml.safe_dump({'options': options}, sort_keys=False)
        return match.group(1) + textwrap.indent(dumped, '    ')

    return DIRECTIVE.sub(move, markdown)


def unshadow_solvers(worktree):
    """Remove the solver subpackages named like the solver they implement.

    Up to v0.2.3, e.g. `dynamiqs.integrators.sesolve` is both a function and a
    subpackage, and current griffe resolves it to the subpackage. The doctests have
    already run, and mkdocstrings only analyses the source statically, so the
    subpackages can be removed from the worktree.
    """
    integrators = worktree / 'dynamiqs' / 'integrators'
    for path in integrators.iterdir():
        if path.is_dir() and (integrators / 'apis' / f'{path.name}.py').exists():
            shutil.rmtree(path)


def overlay_theme(worktree, shim_dir):
    """Generate the release API pages, then replace its MkDocs setup by the current
    Zensical one.
    """
    run(
        sys.executable, 'docs/generate_python_api.py', cwd=worktree,
        env={**os.environ, 'PYTHONPATH': str(shim_dir)},
    )  # fmt: skip

    for markdown_file in (worktree / 'docs').rglob('*.md'):
        markdown = markdown_file.read_text()
        markdown = move_custom_options_to_extra(pattern.sub('', markdown))
        markdown_file.write_text(markdown)

    for path in THEME_FILES:
        source, target = ROOT / path, worktree / path
        if source.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(source, target)
        else:
            shutil.copy(source, target)

    with (ROOT / 'zensical.toml').open('rb') as f:
        config = tomllib.load(f)
    with (worktree / 'mkdocs.yml').open() as f:
        config['project']['nav'] = yaml.load(f, _IgnorePythonTagsLoader)['nav']
    with (worktree / 'zensical.toml').open('wb') as f:
        tomli_w.dump(config, f)

    (worktree / 'mkdocs.yml').unlink()
    (worktree / 'docs' / 'hooks.py').unlink(missing_ok=True)


def main():
    tags = sys.argv[1:] or release_tags()
    failed_doctests = []
    with tempfile.TemporaryDirectory() as tmp:
        shim_dir = Path(tmp, 'shim')
        shim_dir.mkdir()
        (shim_dir / 'mkdocs_gen_files.py').write_text(GEN_FILES_SHIM)

        for tag in tags:
            print(f'\n>>> {tag}', file=sys.stderr, flush=True)
            worktree = Path(tmp, tag)
            run('git', 'worktree', 'add', '--detach', worktree, tag, cwd=ROOT)
            try:
                if not run_doctests(worktree, tag):
                    failed_doctests.append(tag)
                unshadow_solvers(worktree)
                overlay_theme(worktree, shim_dir)
                run('mike', 'deploy', tag.removeprefix('v'), cwd=worktree)
            finally:
                run('git', 'worktree', 'remove', '--force', worktree, cwd=ROOT)

    if failed_doctests:
        print(
            f'\nSome doctests failed for {", ".join(failed_doctests)}: the figures of '
            'the failing examples are missing from these versions.',
            file=sys.stderr,
        )
    print(
        '\nTo make the latest release the stable one and remove the minor-level '
        'versions:\n'
        f'    mike alias --update-aliases {tags[-1].removeprefix("v")} stable\n'
        '    mike delete 0.3 0.2\n'
        '    mike set-default stable\n'
        'Then review with `mike serve` and publish with `git push origin gh-pages`.',
        file=sys.stderr,
    )


if __name__ == '__main__':
    main()
