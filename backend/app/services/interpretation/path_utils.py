"""Case-insensitive path resolution for Windows paths accessed via WSL.

Windows file systems are case-insensitive, but WSL (Windows Subsystem for
Linux) treats them as case-sensitive. This module walks the directory tree
and matches each path component against the actual filesystem casing so that
files referenced with different capitalizations resolve to the same path.
"""

from __future__ import annotations

from pathlib import Path


def canonicalize_existing_path(path: Path) -> Path:
    """Resolve an existing path while preserving the filesystem's actual casing.

    This matters for transcript grouping because reviewed corpora may store the
    same Windows path with different letter casing under WSL.
    """

    if not path.is_absolute():
        return path

    current = Path(path.anchor)
    for part in path.parts[1:]:
        try:
            entry_names = {child.name.lower(): child.name for child in current.iterdir()}
        except OSError:
            current = current / part
            continue
        current = current / entry_names.get(part.lower(), part)
    return current
