"""
Logging utilities
"""

import re

from dowhy.gcm.auto import AutoAssignmentSummary


def process_summary(summary: str | AutoAssignmentSummary) -> dict[str, float]:
    if isinstance(summary, AutoAssignmentSummary):
        summary = str(summary)

    N = 0
    ranking: dict[str, float] = {}

    for node in summary.split('\n\n'):
        node = re.sub(r'\n +', ' ', node)
        mechanisms = [
            mech.strip().split(': ')[0]
            for mech in re.findall(
                r'\n.+: -?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?',
                node
            )
        ]

        if not mechanisms:
            continue

        N += 1

        for idx, mech in enumerate(mechanisms):
            if mech not in ranking:
                ranking[mech] = 0

            ranking[mech] += idx + 1

    for mech in ranking:
        ranking[mech] /= N

    ranking = dict(sorted(ranking.items(), key=lambda item: item[1]))
    return ranking
