"""The extraction plugins.

Importing this package is what populates the registry: each extractor module
registers itself on import, so `get_extractor` can only see a tool that has been
imported here. Adding a plugin is adding a module and a line below.
"""

from .base import BaseExtractor, get_extractor, register_extractor
from .extractors import cicflowmeter, zeek        # noqa: F401 -- import registers

__all__ = ["BaseExtractor", "get_extractor", "register_extractor"]
