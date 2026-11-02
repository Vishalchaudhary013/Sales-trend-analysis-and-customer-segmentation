"""Optional local overrides from a ``.env`` file and environment variables.

Precedence (highest first): real environment variables → ``.env`` in the
project root → the defaults written in :mod:`src.config`. With no ``.env`` and
no variables set, nothing changes.

The parser is a small standard-library implementation (no extra dependency):
``KEY=value`` lines, optional ``export`` prefix, ``#`` comments, blank lines
and single/double-quoted values. Values are validated with clear messages.

Set ``SALES_DASHBOARD_IGNORE_DOTENV=1`` to skip the ``.env`` file (the test
suite does this so personal settings never change test expectations).
"""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Callable
from pathlib import Path

IGNORE_FLAG = "SALES_DASHBOARD_IGNORE_DOTENV"
_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


class SettingsError(ValueError):
    """Raised when a setting has an invalid value (message names the variable)."""


# Problems found while reading settings; the app shows them on every page.
ERRORS: list[str] = []


def report(message: str) -> None:
    """Record a settings problem and emit a warning (shown by scripts on stderr)."""
    ERRORS.append(message)
    warnings.warn(f"{message} Using the default instead.", stacklevel=3)


def safe(reader: Callable[..., object], name: str, default: object, *args: object) -> object:
    """Call ``reader(name, default, *args)``; on an invalid value report it, return default."""
    try:
        return reader(name, default, *args)
    except SettingsError as exc:
        report(str(exc))
        return default


def safe_load_dotenv(path: Path) -> dict[str, str]:
    """Like :func:`load_dotenv`, but a malformed file is reported instead of raising."""
    try:
        return load_dotenv(path)
    except SettingsError as exc:
        report(f"{exc} (in {path.name}); the file was not applied.")
        return {}


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse ``.env`` text into a dict (later keys win; malformed lines are reported).

    Raises:
        SettingsError: For a non-comment line that is not ``KEY=value``.
    """
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            raise SettingsError(f".env line {number} is not KEY=value: {raw!r}")
        key, value = match.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()  # inline comment after unquoted value
        values[key] = value
    return values


def load_dotenv(path: Path) -> dict[str, str]:
    """Load ``path`` into ``os.environ`` without overriding variables already set.

    Returns:
        The key/values that were applied (empty if the file is absent or ignored).
    """
    if os.environ.get(IGNORE_FLAG) or not path.is_file():
        return {}
    applied = {}
    for key, value in parse_dotenv(path.read_text(encoding="utf-8")).items():
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


def _raw(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else None


def env_int(name: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    """Read an integer setting with optional inclusive bounds."""
    raw = _raw(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a whole number, got {raw!r}.") from exc
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise SettingsError(f"{name} must be between {minimum} and {maximum}, got {value}.")
    return value


def env_float(
    name: str, default: float, minimum: float | None = None, maximum: float | None = None
) -> float:
    """Read a decimal setting with optional exclusive bounds."""
    raw = _raw(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a number, got {raw!r}.") from exc
    if (minimum is not None and value <= minimum) or (maximum is not None and value >= maximum):
        raise SettingsError(f"{name} must be between {minimum} and {maximum}, got {value}.")
    return value


def env_path(name: str, default: Path, root: Path) -> Path:
    """Read a path setting; relative values resolve against ``root`` (the project)."""
    raw = _raw(name)
    if raw is None:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else root / path


def env_choice(name: str, default: str, choices: list[str] | tuple[str, ...]) -> str:
    """Read a setting restricted to ``choices`` (case-insensitive, returned as listed)."""
    raw = _raw(name)
    if raw is None:
        return default
    lookup = {c.upper(): c for c in choices}
    if raw.upper() not in lookup:
        raise SettingsError(f"{name} must be one of {', '.join(choices)}; got {raw!r}.")
    return lookup[raw.upper()]
