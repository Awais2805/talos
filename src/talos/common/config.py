"""Load config.yml and turn it into resolved lake locations.

One object answers "where does zone X for dataset Y live", so no script has to
carry its own bucket name or prefix again. The same config drives a local
directory and an S3 bucket -- only `lake.root` changes.

What is NOT here: dataset roles. Whether a dataset may train is a property of
an experiment, not of the lake -- see `talos.experiment`.
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

        # One shape: lake.root plus an optional lake.zones override. `zones` is
        # for a lake whose layout this code did not create -- it is not how a
        # Talos lake is expected to look, and omitting it is the normal case.
        self.root = zones.normalise_root(lake.get("root") or "./lake")
        self.zone_templates = {**zones.DEFAULT_TEMPLATES, **(lake.get("zones") or {})}
        self.region = aws.get("region")
        self.datasets = self.doc.get("datasets") or {}
        self.extractor = self.doc.get("extractor", "zeek")
        self.model = self.doc.get("model", "xgboost")
        self.installed: list[str] = []      # filled by install_plugins()

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
        config = cls(doc, source=path)
        # Installed HERE rather than by each command, because a config that names
        # a plug-in and does not get it is worse than one that names none: some
        # code paths would use the user's extractor and others silently would not.
        config.install_plugins()
        return config

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
                          region=self.region, profile=self.resources)

    def zone(self, name: str, dataset: str | None = None, **scope) -> str:
        """Full URI of a zone, optionally scoped to one dataset.

        The source is taken from the dataset's own declaration unless the caller
        overrides it, so no call site has to remember that CIC datasets are
        `datasets` and a netem run is not.
        """
        if name not in self.zone_templates:
            known = ", ".join(sorted(self.zone_templates))
            raise ConfigError(f"unknown zone {name!r}; known zones: {known}")
        scope.setdefault("source", self.source_of(dataset) if dataset else None)
        return zones.resolve(self.root, self.zone_templates[name], dataset, **scope)

    def source_of(self, dataset: str) -> str:
        """Which of the three kinds of place this dataset's traffic came from.

        Declared per dataset in config.yml; defaults to `datasets`, since a
        public IDS corpus is what an unannotated entry almost always is. The
        value matters beyond filing: a honeypot capture has no attack schedule,
        so the schedule labeller must never be pointed at one.
        """
        source = (self.datasets.get(dataset) or {}).get("source", zones.DEFAULT_SOURCE)
        if source not in zones.SOURCES:
            raise ConfigError(
                f"dataset {dataset!r} declares source {source!r}; "
                f"known sources: {', '.join(zones.SOURCES)}")
        return source

    def datasets_from_source(self, source: str) -> list[str]:
        return sorted(d for d in self.datasets if self.source_of(d) == source)

    def parquet_glob(self, zone: str, dataset: str) -> str:
        """Every parquet under one dataset's zone, at any depth."""
        return f"{self.zone(zone, dataset)}/**/*.parquet"

    # -------------------------------------------------------------- plug-ins

    def install_plugins(self) -> list[str]:
        """Apply the `plugins:` block: the user's own extractors, manifests, methods.

        Keyed by plug-in kind, so a directory of manifests is never searched for
        experiments:

            plugins:
              extractor:  [~/talos/nprobe.py]     # a .py, or a dir of them
              manifest:   [~/talos/manifests]     # a directory of .yaml
        """
        import talos.points                     # noqa: F401 -- populates POINTS
        from talos.common.plugins import PluginError, install

        try:
            self.installed = install(self.doc.get("plugins"))
        except PluginError as exc:
            raise ConfigError(f"{self.source or 'config'}: {exc}") from exc
        return self.installed

    @property
    def reports(self) -> Path:
        return Path((self.doc.get("reports") or {}).get("dir", reports_dir()))

    @property
    def eda_dir(self) -> Path:
        return self.reports / "eda"

    # ------------------------------------------------------------ resources

    @property
    def resources(self):
        """What this machine may spend, detected unless config.yml says otherwise.

        The lake root is handed to detection so spill can land beside the data
        when the lake is local -- usually the big disk rather than the one
        holding the operating system.
        """
        from talos.common.resources import ResourceProfile
        return ResourceProfile.from_config(
            self.doc.get("resources"),
            lake_root=None if self.is_remote else self.root)

    # ---------------------------------------------------------------- roles
    #
    # There are none here, deliberately. "May this dataset contribute to a
    # training pool" is a property of a STUDY, not of the data: 2019 is a
    # holdout for the cross-domain experiment and another experiment could
    # legitimately train on it. Roles live in an `ExperimentSpec`, pinned inside
    # `experiment_sha` beside the result they produced. See talos.experiment.

    def describe(self) -> str:
        lines = [
            f"config    {self.source or '(defaults)'}",
            f"lake      {self.root}  ({'remote' if self.is_remote else 'local'})",
            f"reports   {self.reports}",
            f"extractor {self.extractor}",
        ]
        lines.append(self.resources.describe())
        if self.is_remote and self.region:
            lines.append(f"region    {self.region}")
        lines.append("zones:")
        for zone in zones.ZONE_ORDER:
            template = self.zone_templates[zone]
            if zones.is_source_scoped(template):
                # Shown per source rather than truncated at `sources/`, because
                # "where do my pcaps go" is the question this command exists for.
                for source in zones.SOURCES:
                    lines.append(f"  {zone:<10} {source:<9} "
                                 f"{zones.resolve(self.root, template, source=source)}")
            else:
                lines.append(f"  {zone:<10} {'(all)':<9} "
                             f"{zones.resolve(self.root, template)}")
        if self.datasets:
            lines.append("datasets:")
            for name in sorted(self.datasets):
                lines.append(f"  {name:<16} {self.source_of(name)}")

        import talos.points                     # noqa: F401 -- populates POINTS
        from talos.common.plugins import POINTS
        lines.append("plug-ins:")
        for kind in sorted(POINTS):
            lines.append(f"  {POINTS[kind].describe()}")
        for line in self.installed:
            lines.append(f"  + {line}")
        return "\n".join(lines)
