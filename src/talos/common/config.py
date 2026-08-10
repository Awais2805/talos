"""Load config.yml and turn it into resolved lake locations.

One object answers "where does zone X for dataset Y live", so no script has to
carry its own bucket name or prefix again. The same config drives a local
directory and an S3 bucket -- only `lake.root` changes.

The legacy shape (`aws.bucket` plus a flat `lake:` map of prefixes) is still
accepted and normalised, so an existing bucket keeps working untouched.
"""

from pathlib import Path

import yaml

from talos.common import zones
from talos.common.paths import default_config, reports_dir


class ConfigError(Exception):
    pass


class Config:

    def __init__(self, doc: dict, source: Path | None = None):
        self.doc = doc or {}
        self.source = source

        lake = self.doc.get("lake") or {}
        aws = self.doc.get("aws") or {}

        # Current shape: lake.root + lake.zones. Legacy shape: aws.bucket and a
        # flat lake: map whose values are bare prefixes.
        root = lake.get("root")
        templates = lake.get("zones")
        if root is None and aws.get("bucket"):
            root = f"s3://{aws['bucket']}"
        if templates is None:
            legacy = {k: v for k, v in lake.items()
                      if k not in ("root", "zones") and isinstance(v, str)}
            templates = {k: (v if "{dataset}" in v else f"{v}/{{dataset}}")
                         for k, v in legacy.items()} or None

        self.root = zones.normalise_root(root or "./lake")
        self.zone_templates = {**zones.DEFAULT_TEMPLATES, **(templates or {})}
        self.region = aws.get("region")
        self.datasets = self.doc.get("datasets") or {}
        self.extractor = self.doc.get("extractor", "zeek")
        self.model = self.doc.get("model", "xgboost")

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, path=None) -> "Config":
        """Read config from an explicit path, $TALOS_CONFIG, or ./config.yml."""
        path = Path(path) if path else default_config()
        if not path.exists():
            raise ConfigError(
                f"no config at {path}. Run `talos init` to create a lake, or point "
                f"at one with --config / $TALOS_CONFIG.")
        try:
            doc = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
        return cls(doc, source=path)

    # ------------------------------------------------------------- locations

    @property
    def is_remote(self) -> bool:
        return zones.is_remote(self.root)

    def lake(self):
        """The lake this config describes.

        The only way a LakeClient should be built: constructing one by hand
        loses `lake.zones` and silently falls back to the default layout, which
        resolves to real-looking paths that hold no data.
        """
        from talos.common.lake.lake import LakeClient
        return LakeClient(root=self.root, templates=self.zone_templates,
                          region=self.region)

    def zone(self, name: str, dataset: str | None = None, **scope) -> str:
        """Full URI of a zone, optionally scoped to one dataset."""
        if name not in self.zone_templates:
            known = ", ".join(sorted(self.zone_templates))
            raise ConfigError(f"unknown zone {name!r}; known zones: {known}")
        return zones.resolve(self.root, self.zone_templates[name], dataset, **scope)

    def parquet_glob(self, zone: str, dataset: str) -> str:
        """Every parquet under one dataset's zone, at any depth."""
        return f"{self.zone(zone, dataset)}/**/*.parquet"

    @property
    def reports(self) -> Path:
        return Path((self.doc.get("reports") or {}).get("dir", reports_dir()))

    @property
    def eda_dir(self) -> Path:
        return self.reports / "eda"

    # ---------------------------------------------------------------- roles

    def role(self, dataset: str) -> str:
        """train | holdout | unknown. Governs whether a dataset may be trained on."""
        return (self.datasets.get(dataset) or {}).get("role", "unknown")

    def datasets_with_role(self, role: str) -> list[str]:
        return sorted(d for d in self.datasets if self.role(d) == role)

    def describe(self) -> str:
        lines = [
            f"config    {self.source or '(defaults)'}",
            f"lake      {self.root}  ({'remote' if self.is_remote else 'local'})",
            f"reports   {self.reports}",
            f"extractor {self.extractor}",
        ]
        if self.is_remote and self.region:
            lines.append(f"region    {self.region}")
        lines.append("zones:")
        for zone in zones.ZONE_ORDER:
            lines.append(f"  {zone:<10} {self.zone(zone)}")
        if self.datasets:
            lines.append("datasets:")
            for name in sorted(self.datasets):
                lines.append(f"  {name:<16} {self.role(name)}")
        return "\n".join(lines)
