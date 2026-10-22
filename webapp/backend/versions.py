"""
Pipeline versions: which steps each version enables. The backend owns this list; the frontend
only sends a version_id. Versions that aren't listed are rejected, never run as the baseline.

The frontend also lists claim-cache, two-ai, two-caches and parallel: add them here once implemented.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Version:
    id: str
    caches: tuple = ()        # "claims", "evidence"
    extraction: str = "rules"  # "rules" or "model"
    parallel: bool = False


VERSIONS = {v.id: v for v in [
    Version("baseline"),   # v0: no cache, rule-based extraction, sequential
]}


def get_version(version_id) -> Version | None:
    return VERSIONS.get(version_id)
