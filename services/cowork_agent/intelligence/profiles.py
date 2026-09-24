"""The intelligence profiles an agent can run a request with.

``config/agents/<name>/intelligence.json``::

    {"schema": 1,
     "default": {"model": null, "effort": null},
     "efforts": ["low", "medium", "high", "xhigh", "max"],
     "profiles": [
       {"id": "deep", "model": null, "effort": "high",
        "use_when": "features, debugging or refactors across several parts of the codebase"}]}

- ``default`` is the fallback setup, used when nothing picks a profile.
  ``null`` means "pass no flag": the agent's own configuration decides.
- ``efforts`` lists the effort levels this agent accepts. Core code checks
  profiles and requests against it, so it never needs to know the agent.
- ``profiles`` are what a request (or the decision model) chooses from. Each
  ``use_when`` is the plain-English description the choice is made against.

The file is re-read when it changes, so editing it changes the choice with no
code change and no restart. A missing file means the agent does not use
profiles; an invalid one is logged once and treated as missing, never raised
into a chat.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]  # intelligence/ → cowork_agent/ → services/ → repo root
_AGENTS_DIR = _REPO_ROOT / "config" / "agents"

FILENAME = "intelligence.json"
SCHEMA = 1

#: The decision model's "none of these fits" option; no profile may take it.
UNKNOWN = "unknown"

_AGENT_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
_PROFILE_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
# Model ids and aliases: ``sonnet``, ``claude-opus-5-5``, ``claude-opus-5-5[1m]``.
# The first character is never ``-``, so a value cannot become a CLI flag.
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:\[\]-]{0,127}")
_EFFORT = re.compile(r"[a-z][a-z0-9_-]{0,31}")
_USE_WHEN_MAX = 300
# A decision-model choice takes at most 120 options, one of them ``unknown``.
_MAX_PROFILES = 119


class ProfileError(ValueError):
    """``intelligence.json`` does not describe a usable set of profiles."""


@dataclass(frozen=True)
class Setup:
    """What to run with. ``None`` passes no flag for that field."""

    model: str | None
    effort: str | None


@dataclass(frozen=True)
class Profile:
    id: str
    model: str | None
    effort: str | None
    use_when: str

    @property
    def setup(self) -> Setup:
        return Setup(model=self.model, effort=self.effort)


@dataclass(frozen=True)
class IntelligenceConfig:
    default: Setup
    efforts: tuple[str, ...]
    profiles: tuple[Profile, ...]
    #: sha256 of the file's bytes, so a logged decision names the version it used.
    sha256: str = ""

    def profile(self, profile_id: str) -> Profile | None:
        for profile in self.profiles:
            if profile.id == profile_id:
                return profile
        return None

    @property
    def profile_ids(self) -> tuple[str, ...]:
        return tuple(p.id for p in self.profiles)


def is_valid_model(value: object) -> bool:
    return isinstance(value, str) and bool(_MODEL.fullmatch(value))


def _optional_model(value: object, where: str) -> str | None:
    if value is None:
        return None
    if not is_valid_model(value):
        raise ProfileError(f"{where}: model {value!r} is not a model id")
    return value


def _optional_effort(value: object, efforts: tuple[str, ...], where: str) -> str | None:
    if value is None:
        return None
    if value not in efforts:
        raise ProfileError(f"{where}: effort {value!r} is not one of {', '.join(efforts)}")
    return value


def parse(document: object, *, sha256: str = "") -> IntelligenceConfig:
    """Check an ``intelligence.json`` document. Raises :class:`ProfileError`."""
    if not isinstance(document, dict):
        raise ProfileError("the document is not a JSON object")
    if document.get("schema") != SCHEMA:
        raise ProfileError(f"schema is {document.get('schema')!r}, expected {SCHEMA}")

    efforts = document.get("efforts")
    if (not isinstance(efforts, list) or not efforts
            or not all(isinstance(e, str) and _EFFORT.fullmatch(e) for e in efforts)):
        raise ProfileError("efforts must be a non-empty list of effort names")
    efforts = tuple(efforts)

    default = document.get("default")
    if not isinstance(default, dict):
        raise ProfileError("default must be an object with model and effort")
    default_setup = Setup(
        model=_optional_model(default.get("model"), "default"),
        effort=_optional_effort(default.get("effort"), efforts, "default"),
    )

    raw_profiles = document.get("profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ProfileError("profiles must be a non-empty list")
    if len(raw_profiles) > _MAX_PROFILES:
        raise ProfileError(f"at most {_MAX_PROFILES} profiles are allowed")

    profiles: list[Profile] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_profiles):
        if not isinstance(raw, dict):
            raise ProfileError(f"profiles[{index}] is not an object")
        pid = raw.get("id")
        if not isinstance(pid, str) or not _PROFILE_ID.fullmatch(pid):
            raise ProfileError(f"profiles[{index}]: id {pid!r} must match {_PROFILE_ID.pattern}")
        if pid == UNKNOWN:
            raise ProfileError(f"profiles[{index}]: {UNKNOWN!r} is reserved for the decision model")
        if pid in seen:
            raise ProfileError(f"profiles[{index}]: id {pid!r} is used twice")
        seen.add(pid)
        use_when = raw.get("use_when")
        if not isinstance(use_when, str) or not use_when.strip():
            raise ProfileError(f"profile {pid!r}: use_when must describe when to use it")
        if len(use_when) > _USE_WHEN_MAX:
            raise ProfileError(f"profile {pid!r}: use_when is longer than {_USE_WHEN_MAX} characters")
        profiles.append(Profile(
            id=pid,
            model=_optional_model(raw.get("model"), f"profile {pid!r}"),
            effort=_optional_effort(raw.get("effort"), efforts, f"profile {pid!r}"),
            use_when=use_when.strip(),
        ))

    return IntelligenceConfig(
        default=default_setup, efforts=efforts, profiles=tuple(profiles), sha256=sha256,
    )


def config_path(agent_name: str) -> Path | None:
    """``config/agents/<agent_name>/intelligence.json``, or ``None`` for a name
    that is not a single safe path segment (it can come from a request body)."""
    if not isinstance(agent_name, str) or not _AGENT_NAME.fullmatch(agent_name):
        return None
    return _AGENTS_DIR / agent_name / FILENAME


# agent_name -> ((path, mtime_ns, size), config or None)
_cache: dict[str, tuple[tuple[Path, int, int], IntelligenceConfig | None]] = {}


def load(agent_name: str) -> IntelligenceConfig | None:
    """The agent's profiles, or ``None`` when it has none (or they are invalid)."""
    path = config_path(agent_name)
    if path is None:
        return None
    try:
        st = path.stat()
    except OSError:
        _cache.pop(agent_name, None)
        return None
    stamp = (path, st.st_mtime_ns, st.st_size)
    cached = _cache.get(agent_name)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        raw = path.read_bytes()
        config: IntelligenceConfig | None = parse(
            json.loads(raw), sha256=hashlib.sha256(raw).hexdigest(),
        )
    except (OSError, ValueError) as exc:  # JSONDecodeError and ProfileError are ValueErrors
        log.warning("intelligence: %s is not usable, so %s runs without profiles: %s",
                    path, agent_name, exc)
        config = None
    _cache[agent_name] = (stamp, config)
    return config
