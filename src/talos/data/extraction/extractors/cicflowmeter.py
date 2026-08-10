# src/talos/data/extraction/cicflowmeter.py
import logging
from talos.data.extraction.base import BaseExtractor, register_extractor

logger = logging.getLogger(__name__)

@register_extractor
class CICFlowMeterExtractor(BaseExtractor):
    def __init__(self, version: str = "4.0"):
        self._version = version

    @property
    def name(self) -> str:
        return "cicflowmeter"

    @property
    def version(self) -> str:
        return self._version

    def extract(self, pcap_paths: list[str], output_dir: str) -> None:
        """
        Extracts the full 84-statistical feature set from pcaps using CICFlowMeter.
        (e.g., flow duration, inter-arrival times, active packet flags, subflow stats).
        """
        logger.error("CICFlowMeter extraction is not yet implemented.")
        raise NotImplementedError(
            "CICFlowMeter extractor boilerplate. Implement Java JAR invocation or "
            "Python wrapper here to extract the complete 84-feature statistical flow set."
        )