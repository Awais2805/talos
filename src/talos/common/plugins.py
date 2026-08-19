"""Bring your own.

Accepts:

    a .py with a class in it   ->  CodePlugins     imported to exist
    a .yaml file               ->  ConfigPlugins   found and read

The plug-in points, and where each is created:

    KIND                     SUPPLIES   CREATED IN
    extractor                .py        talos.data.extraction     [gate 7]
    manifest                 .yaml      talos.data.labelling      [gate 7]
    experiment               .yaml      talos.experiment          [gate 7]
    labelling method         .py        talos.data.labelling      [gate 8]
    method, label space      .yaml      talos.data.labelling      [gate 9]
    encoder decoder head     .py        talos.parts                [gate 11]
    model                    .py        talos.engine              [later]

Adding a point is one line -- `X = CodePlugins("thing", error=ThingError)` -- and
it is addressable from config.yml immediately, because a point registers itself
under its kind.

"""

from __future__ import annotations

import importlib.util
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence, Type

import yaml

from talos.common.provenance import Sha12, content_sha, digest


class PluginError(Exception):
    """Default for a point not given its stage's own exception type."""


# --------------------------------------------------------------------- origin

BUILT_IN = "built-in"     # shipped with the package
SEARCHED = "searched"     # found in a directory the user added
EXTERNAL = "external"     # named directly, e.g. --manifest ~/mine.yaml


@dataclass(frozen=True)
class Origin:
    """Which plug-in was used, and whose it was."""

    kind: str
    name: str
    where: str                    # BUILT_IN | SEARCHED | EXTERNAL
    source: Path | None = None

    @property
    def built_in(self) -> bool:
        return self.where == BUILT_IN

    def describe(self) -> str:
        """Quiet for ours, loud for theirs."""
        if self.built_in:
            return f"{self.kind:<12} {self.name}"
        return f"{self.kind:<12} {self.name}  [{self.where} {self.source}]"


# ---------------------------------------------------------------- the points

#: Every plug-in point, by kind. Lets config.yml name a kind without this module
#: knowing the list.
POINTS: dict[str, "PluginPoint"] = {}


class PluginPoint(ABC):
    """One replaceable thing, whichever kind of file the user brings.

    Holds only what both kinds share: the kind's name, the exception the owning
    stage wants raised, and the single unknown-name refusal.
    """

    def __init__(self, kind: str, error: Type[Exception] = PluginError,
                 *, addressable: bool = True):
        """`addressable` publishes the point under `kind` for config.yml.

        True for the module-level point a stage owns; False for a throwaway --
        a test, or a loader told to read one specific directory. It defaults to
        True because forgetting it there would silently make a point unreachable
        from config.yml, whereas forgetting False is a loud collision.
        """
        self.kind = kind
        self.error = error
        if addressable:
            if kind in POINTS:
                raise PluginError(
                    f"two plug-in points both call themselves {kind!r}. The kind "
                    f"is how config.yml addresses a point, so it must be unique.")
            POINTS[kind] = self

    # -------------------------------------------------------------- contract

    @abstractmethod
    def names(self) -> tuple[str, ...]:
        """Every name that resolves, in the order they would be found."""

    @abstractmethod
    def origin_of(self, name: str) -> Origin:
        """Where the plug-in with this name comes from."""

    @abstractmethod
    def add(self, path: str | Path) -> tuple[str, ...]:
        """Take on a user-supplied file or directory; returns what appeared.

        One verb for both kinds: from config.yml they are the same gesture.
        """

    # ---------------------------------------------------------------- shared

    def refuse(self, name: str, detail: str = "") -> NoReturn:
        """The one unknown-name message, for both kinds."""
        have = ", ".join(self.names()) or "none"
        raise self.error(
            f"unknown {self.kind} {name!r}{' ' + detail if detail else ''}. "
            f"Have: {have}")

    def __contains__(self, name: object) -> bool:
        return name in self.names()

    def __len__(self) -> int:
        return len(self.names())

    def describe(self) -> str:
        return f"{self.kind:<18} {', '.join(self.names()) or '(none)'}"


# ------------------------------------------------------- code: name -> class

class CodePlugins(PluginPoint):
    """A `.py` with a class in it.

    The name is read off the CLASS, never off an instance. An earlier version
    conjured one to read it, so any plugin needing a constructor argument broke
    the import of the whole package.
    """

    def __init__(self, kind: str, error: Type[Exception] = PluginError,
                 *, addressable: bool = True):
        super().__init__(kind, error, addressable=addressable)
        self._classes: dict[str, type] = {}
        self._origins: dict[str, Origin] = {}
        self._loaded: set[Path] = set()

    # ------------------------------------------------------------ populating

    def register(self, cls: type) -> type:
        """Decorator. Registers `cls` under its class-level `name`."""
        name = getattr(cls, "name", "")
        if not isinstance(name, str) or not name:
            raise self.error(
                f"{cls.__name__} must declare a class-level `name`, e.g. "
                f'`name = "mine"`. It is read off the class at import time.')
        if name in self._classes and self._classes[name] is not cls:
            raise self.error(
                f"two {self.kind}s both call themselves {name!r}: "
                f"{self._classes[name].__name__} and {cls.__name__}")
        self._classes[name] = cls
        self._origins.setdefault(name, Origin(self.kind, name, BUILT_IN))
        return cls

    def add(self, path: str | Path) -> tuple[str, ...]:
        """Import the user's `.py`, or every `.py` in their directory.

        This EXECUTES the file -- there is no way to add a class to a registry
        without running the code that defines it. The path comes from whoever ran
        the command or wrote the config, never from data in the lake.

        Re-adding a loaded file is a no-op: without that, installing twice in one
        process re-imports the module and trips the collision refusal on the
        user's own plugin.
        """
        path = Path(path).expanduser().resolve()
        if path.is_dir():
            found: list[str] = []
            for child in sorted(path.glob("*.py")):
                if not child.name.startswith("_"):
                    found.extend(self.add(child))
            return tuple(found)

        if not path.exists():
            raise self.error(f"no {self.kind} plug-in at {path}")
        if path in self._loaded:
            return ()
        return self._import(path)

    def _import(self, path: Path) -> tuple[str, ...]:
        before = set(self._classes)
        module_name = f"_talos_plugin_{digest(str(path).encode())[:8]}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise self.error(f"{path} is not importable as a Python module")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:                          # noqa: BLE001
            del sys.modules[module_name]
            raise self.error(
                f"{path} failed to import: {type(exc).__name__}: {exc}") from exc

        added = tuple(sorted(set(self._classes) - before))
        if not added:
            # Importing something inert and carrying on is how a run silently
            # uses the shipped plugin while its operator believes otherwise.
            raise self.error(
                f"{path} imported but registered no {self.kind}. The class must "
                f"be decorated with the {self.kind} point's `register`.")
        self._loaded.add(path)
        for name in added:
            self._origins[name] = Origin(self.kind, name, EXTERNAL, path)
        return added

    # ------------------------------------------------------------- resolving

    def cls_for(self, name: str) -> type:
        """The class, unconstructed -- for reading class attributes."""
        if name not in self._classes:
            self.refuse(name)
        return self._classes[name]

    def get(self, name: str, **kwargs) -> Any:
        """Resolve and construct."""
        return self.cls_for(name)(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._classes))

    def origin_of(self, name: str) -> Origin:
        if name not in self._origins:
            self.refuse(name)
        return self._origins[name]


# ------------------------------------------------ config: name -> declaration

@dataclass(frozen=True)
class Declaration:
    """A resolved YAML declaration: parsed, hashed, attributed."""

    origin: Origin
    path: Path
    doc: Mapping[str, Any]
    sha: Sha12

    @property
    def name(self) -> str:
        return self.origin.name

    @property
    def kind(self) -> str:
        return self.origin.kind

    @property
    def built_in(self) -> bool:
        return self.origin.built_in

    def describe(self) -> str:
        return f"{self.origin.describe()}  sha {self.sha}"


class ConfigPlugins(PluginPoint):
    """A `.yaml` file.

    Two layouts, because the package already uses both: manifests are
    `<dir>/<name>.yaml`, experiments are `<dir>/<name>/experiment.yaml`.
    `filename` picks.

    Added directories are searched BEFORE the built-in one -- an override is
    meant to win. `Origin` is what makes that safe rather than forbidding it.
    """

    SUFFIXES = (".yaml", ".yml")

    def __init__(self, kind: str,
                 built_in: str | Path | Sequence[str | Path] | None = None,
                 filename: str | None = None,
                 error: Type[Exception] = PluginError,
                 *, addressable: bool = True):
        """`built_in` may name several directories.

        Method declarations ship beside the code that reads them, and that code
        is in more than one package -- `schedule/method.yaml` and
        `behavioural/ae/method.yaml`. All of them are equally shipped, so all
        of them report as BUILT_IN.
        """
        super().__init__(kind, error, addressable=addressable)
        if built_in is None:
            self.built_in_dirs: tuple[Path, ...] = ()
        elif isinstance(built_in, (str, Path)):
            self.built_in_dirs = (Path(built_in),)
        else:
            self.built_in_dirs = tuple(Path(d) for d in built_in)
        self.filename = filename
        self._added: list[Path] = []

    @property
    def built_in(self) -> Path | None:
        """The first shipped directory, for the common single-directory case."""
        return self.built_in_dirs[0] if self.built_in_dirs else None

    def add(self, path: str | Path) -> tuple[str, ...]:
        """Also look in this directory, ahead of the built-in one."""
        directory = Path(path).expanduser()
        if not directory.is_dir():
            raise self.error(
                f"{directory} is not a directory. A {self.kind} search path is a "
                f"directory of declarations; name a single file at the point of use.")
        if directory not in self._added:
            self._added.append(directory)
        return tuple(self._names_in(directory))

    # ------------------------------------------------------------- searching

    def _dirs(self) -> tuple[tuple[Path, str], ...]:
        pairs = [(d, SEARCHED) for d in self._added]
        pairs += [(d, BUILT_IN) for d in self.built_in_dirs]
        return tuple(pairs)

    def _names_in(self, directory: Path) -> list[str]:
        if not directory.is_dir():
            return []
        if self.filename:
            found = (p.parent.name for p in directory.glob(f"*/{self.filename}"))
        else:
            found = (p.stem for suffix in self.SUFFIXES
                     for p in directory.glob(f"*{suffix}"))
        return sorted(found)

    def _candidate(self, directory: Path, name: str) -> Path:
        if self.filename:
            return directory / name / self.filename
        return directory / f"{name}{self.SUFFIXES[0]}"

    def _direct(self, name: str | Path) -> Path | None:
        """A name that is really a path to an existing declaration file."""
        candidate = Path(name).expanduser()
        if candidate.suffix in self.SUFFIXES and candidate.is_file():
            return candidate
        return None

    def names(self) -> tuple[str, ...]:
        """Every name that resolves, nearest directory first, de-duplicated."""
        seen: dict[str, None] = {}
        for directory, _where in self._dirs():
            for entry in self._names_in(directory):
                seen.setdefault(entry, None)
        return tuple(seen)

    # ------------------------------------------------------------- resolving

    def origin_of(self, name: str | Path) -> Origin:
        direct = self._direct(name)
        if direct is not None:
            return Origin(self.kind, direct.stem, EXTERNAL, direct)

        for directory, where in self._dirs():
            candidate = self._candidate(directory, str(name))
            if candidate.is_file():
                resolved = candidate.parent.name if self.filename else candidate.stem
                return Origin(self.kind, resolved, where, candidate)

        searched = ", ".join(str(d) for d, _ in self._dirs()) or "nowhere"
        self.refuse(str(name), f"under {searched}")

    def path_for(self, name: str | Path) -> Path:
        source = self.origin_of(name).source
        assert source is not None                # a config plug-in is always a file
        return source

    def get(self, name: str | Path) -> Declaration:
        """Resolve, read, parse and hash. Validation is the caller's."""
        origin = self.origin_of(name)
        path = origin.source
        assert path is not None
        try:
            doc = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise self.error(f"{path} is not valid YAML: {exc}") from exc
        if not isinstance(doc, dict):
            raise self.error(
                f"{path} is a {type(doc).__name__}, not a mapping. A {self.kind} "
                f"declaration is a block of named fields.")
        return Declaration(origin=origin, path=path, doc=doc, sha=content_sha(path))


# ------------------------------------------------------ installing from config

def install(declared: Mapping[str, Any] | None) -> list[str]:
    """Apply a `plugins:` block from config.yml; returns a line per thing added.

    Keyed by kind, so a directory of manifests is never searched for experiments.
    An unknown kind is refused: a typo would otherwise be config that looks
    applied and does nothing.
    """
    lines: list[str] = []
    for kind, paths in (declared or {}).items():
        point = POINTS.get(kind)
        if point is None:
            known = ", ".join(sorted(POINTS)) or "none"
            raise PluginError(
                f"config.yml declares plug-ins for {kind!r}, which is not a "
                f"plug-in point. Known kinds: {known}")
        for path in ([paths] if isinstance(paths, (str, Path)) else paths or []):
            added = point.add(path)
            lines.append(f"{kind:<18} {path}  ({', '.join(added) or 'nothing new'})")
    return lines
