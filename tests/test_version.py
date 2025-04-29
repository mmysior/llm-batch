import re
import subprocess
from pathlib import Path

import pytest


def get_current_version() -> str:
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError(f"{pyproject_path} not found")
    with pyproject_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("version"):
                match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', line)
                if match:
                    return match.group(1)
    raise RuntimeError("Version not found in pyproject.toml")


def get_main_version() -> str:
    # Get the pyproject.toml from main branch
    result = subprocess.run(
        ["git", "show", "origin/main:pyproject.toml"],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        if line.strip().startswith("version"):
            match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', line)
            if match:
                return match.group(1)
    raise RuntimeError("Version not found in main branch pyproject.toml")


@pytest.mark.skipif(
    subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True
    ).returncode
    != 0,
    reason="Not inside a git repository",
)
def test_version_bumped() -> None:
    current_version = get_current_version()
    main_version = get_main_version()
    assert current_version != main_version, (
        f"Version not bumped: still {current_version}"
    )
