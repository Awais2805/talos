import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Type

_EXTRACTOR_REGISTRY = {}

# The provenance sidecar every extraction run drops beside its logs. Named here
# because downstream stages have to recognise it to skip it: it sits in the same
# tree as the logs and would otherwise be converted as if it were one of them.
META_FILENAME = "_extractor_meta.json"

def register_extractor(cls: Type['BaseExtractor']):
    """Decorator to register an extractor class.

    The key is read off an uninitialised instance rather than a constructed one:
    `name` is a property, but running __init__ at import time would make a tool
    that needs a constructor argument break the import of the whole package.
    """
    _EXTRACTOR_REGISTRY[cls.__new__(cls).name] = cls
    return cls

def get_extractor(name: str, **kwargs) -> 'BaseExtractor':
    """Retrieve an instantiated extractor from the registry."""
    if name not in _EXTRACTOR_REGISTRY:
        known = ", ".join(_EXTRACTOR_REGISTRY.keys())
        raise ValueError(f"Unknown extractor '{name}'. Known extractors: {known}")
    return _EXTRACTOR_REGISTRY[name](**kwargs)

class BaseExtractor(ABC):
    """The strict contract every extraction tool must follow."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The core name of the tool (e.g., 'zeek', 'cicflowmeter')."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """The version of the tool (e.g., '8.2.1', '4.0')."""
        pass

    @property
    def feature_space(self) -> str:
        """Dynamically builds the feature space from name and version.
        This dictates the S3/local folder, e.g., 'zeek_v8.2.1'.
        """
        return f"{self.name}_v{self.version}"

    def write_metadata(self, output_dir: str, dataset: str, extra_meta: dict = None):
        """Standardized metadata sidecar dropped into the extracted zone."""
        meta = {
            "dataset": dataset,
            "extractor_tool": self.name,
            "extractor_version": self.version,
            "feature_space": self.feature_space,
            "extracted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if extra_meta:
            meta.update(extra_meta)
            
        out_path = Path(output_dir) / META_FILENAME
        out_path.write_text(json.dumps(meta, indent=2))
        return out_path

    @abstractmethod
    def extract(self, pcap_paths: list[str], output_dir: str) -> None:
        """The tool-specific extraction logic.
        
        pcap_paths: List of local paths to the downloaded raw pcaps.
        output_dir: The local scratch directory where the tool should write its logs.
        """
        pass