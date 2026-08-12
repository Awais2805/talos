"""Every plug-in point, imported so config.yml can address it.

A point exists only once the module declaring it has been imported. Without this,
`plugins: manifest: [...]` would be refused as an unknown kind in any command
that had not already imported the labelling package -- a different answer
depending on which command was run.

Adding a plug-in point: declare it in its stage, then add a line here.
"""

from talos.data.extraction import base as _extractor        # noqa: F401
from talos.data.labelling import taxonomy as _taxonomy         # noqa: F401
from talos.data.labelling.schedule import labeller as _schedule     # noqa: F401
from talos.data.labelling import behavioural as _behavioural       # noqa: F401
from talos.data.labelling import method as _method                  # noqa: F401
from talos.data.labelling import space as _space                    # noqa: F401
from talos.data.labelling.schedule import manifest as _manifest      # noqa: F401
from talos.experiment import loader as _experiment          # noqa: F401

__all__: list[str] = []
