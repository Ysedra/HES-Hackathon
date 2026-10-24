"""
Pipeline versions: which steps each version enables. The backend owns this list; the frontend
only sends a version_id. Versions that aren't listed are rejected, never run as the baseline.

The frontend also lists the planned two-ai and two-caches versions. They remain unavailable
until their backend stages exist; the parallel entry is an explicitly labelled prototype, whose
number of workers each request can set (api_helpers.require_version checks the bounds).
"""

import os
from dataclasses import dataclass

# Claim-checking threads for parallel versions, chosen per run in the GUI. Workers mostly wait on
# the network, so the cap protects the laptop and the search providers (rate limits), not the CPU.
MIN_WORKERS = 1
MAX_WORKERS = min(8, os.cpu_count() or 4)
DEFAULT_WORKERS = min(3, MAX_WORKERS)


@dataclass(frozen=True)
class Version:
    id: str
    caches: tuple = ()        # "claims", "evidence"
    extraction: str = "rules"  # "rules" or "model"
    parallel: bool = False
    workers: int = 1          # used when parallel; a request's config can set it within the bounds above
    source_policy: str = "wikipedia_first"


VERSIONS = {v.id: v for v in [
    Version("baseline"),   # v0: no cache, rule-based extraction, sequential
    Version("claim-cache", caches=("claims",)),  # v1: exact verified-claim cache
    # Parallel prototype: deliberately the same components as V1, so its speed difference
    # measures concurrency rather than mixing in the unfinished V2/V3 components.
    Version("parallel", caches=("claims",), parallel=True, workers=DEFAULT_WORKERS),
]}


def get_version(version_id) -> Version | None:
    return VERSIONS.get(version_id)
