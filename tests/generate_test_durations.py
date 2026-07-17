"""Pytest plugin: generate a synthetic .test_durations file for pytest-split.

Instead of timing a full test run, assign each collected test a weight based on
the `@pytest.mark.run(order=...)` priority marker (TEST_INSTANT=0, TEST_SHORT=1,
TEST_LONG=2). pytest-split only uses durations for *relative* balancing across
CI shards, so rough per-category weights are enough.

Regenerate the file with (collection only, no tests are run):
    task durations
"""

import json
from collections import Counter
from pathlib import Path

# rough relative cost per priority category (seconds, only ratios matter)
WEIGHTS = {
    0: 0.1,  # instant tests
    1: 2.0,  # short tests
    2: 15.0,  # long tests
}


def pytest_collection_finish(session):
    durations = {}
    counts = Counter()
    for item in session.items:
        marker = item.get_closest_marker('run')
        order = marker.kwargs.get('order', 0) if marker else 0
        durations[item.nodeid] = WEIGHTS[order]
        counts[order] += 1

    with Path('.test_durations').open('w') as f:
        json.dump(durations, f, indent=2, sort_keys=True)

    total = sum(durations.values())
    reporter = session.config.pluginmanager.get_plugin('terminalreporter')
    reporter.write_line(
        f'\n[duration_gen] wrote {len(durations)} entries to .test_durations: '
        f'{counts[0]} instant, {counts[1]} short, {counts[2]} long '
        f'(synthetic total {total:.0f}s)'
    )
