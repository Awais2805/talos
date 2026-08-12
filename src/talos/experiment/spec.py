"""What an experiment declares.

Content-hashed, so a result can name the exact declaration that produced it. A
hand-maintained version field drifts from the file it describes; a hash cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

# train    -- may contribute flows to a training pool
# holdout  -- evaluation only; MUST NOT enter a training pool
# audit    -- neither; held back for hand inspection and for the label benchmark
ROLES = ("train", "holdout", "audit")
Role = str

UNASSIGNED = "unassigned"


@dataclass(frozen=True)
class ExperimentSpec:
    """One experiment, declared and hashed."""

    name: str
    lake: str | None
    extractor: str
    labels: str
    datasets: Mapping[str, Mapping[str, Any]]
    sha: str
    path: Path | None = None
    doc: Mapping[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------- roles

    def role(self, dataset: str) -> Role:
        """train | holdout | audit | unassigned.

        A dataset the experiment never mentions is `unassigned`, not `train`.
        Defaulting the other way would let a dataset dropped into the lake join
        a training pool because nobody wrote a line about it.
        """
        return (self.datasets.get(dataset) or {}).get("role", UNASSIGNED)

    def datasets_with_role(self, role: Role) -> list[str]:
        return sorted(d for d in self.datasets if self.role(d) == role)

    def reason(self, dataset: str) -> str | None:
        """Why a dataset carries the role it does. Prose, never parsed.

        Hashed with the rest of the file, which is the point: the argument for
        excluding a dataset travels with the result that excluded it.
        """
        text = (self.datasets.get(dataset) or {}).get("reason")
        return str(text).strip() if text else None

    # ------------------------------------------------------------- reporting

    def describe(self) -> str:
        lines = [
            f"experiment  {self.name}",
            f"sha         {self.sha}",
            f"lake        {self.lake or '(from config)'}",
            f"extractor   {self.extractor}",
            f"labels      {self.labels}",
            "datasets:",
        ]
        for name in sorted(self.datasets):
            lines.append(f"  {name:<16} {self.role(name)}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "sha": self.sha, "lake": self.lake,
            "extractor": self.extractor,
            "labels": self.labels,
            "datasets": {d: {"role": self.role(d)} for d in sorted(self.datasets)},
        }
