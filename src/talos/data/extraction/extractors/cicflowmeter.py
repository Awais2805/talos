"""CICFlowMeter — registered, declared unavailable, not implemented.

It is registered so `talos config` can show it as a known plugin and so the
feature-space naming is settled before anyone writes the Java invocation. It
declares itself unavailable so a run refuses immediately rather than after
fetching several hundred GB of pcaps to local scratch.
"""

from __future__ import annotations

import logging

from talos.data.extraction.base import (
    BYTE_COUNTS, BaseExtractor, ExtractionReport, register_extractor,
)

logger = logging.getLogger(__name__)


@register_extractor
class CICFlowMeterExtractor(BaseExtractor):

    name = "cicflowmeter"

    def __init__(self, version: str = "4.0", jar: str | None = None):
        self._version = version
        self.jar = jar

    @property
    def version(self) -> str:
        return self._version

    def capabilities(self) -> frozenset[str]:
        """Statistical flow records with byte counts, but NO stable per-flow id
        and no connection backbone other logs hang off -- so labels could not be
        propagated the way the Zeek path does. Declared now so the guard exists
        before the implementation does."""
        return frozenset({BYTE_COUNTS})

    def available(self) -> tuple[bool, str]:
        return False, (
            "cicflowmeter is a registered placeholder with no implementation yet. "
            "Implement the JAR invocation in extractors/cicflowmeter.py, or set "
            "`extractor: zeek` in config.yml.")

    def extract(self, pcap_paths: list[str], output_dir: str) -> ExtractionReport:
        """The 84-feature statistical flow set: durations, inter-arrival times,
        flag counts, subflow statistics. Not yet written.

        Note for whoever does write it: these flows are NOT comparable with
        Zeek's. Different boundaries, different counts, different columns. The
        feature space keeps them apart on disk; nothing keeps them apart in an
        analysis that pools them by hand.
        """
        raise NotImplementedError(self.available()[1])
