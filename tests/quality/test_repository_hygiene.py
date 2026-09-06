import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_TRACKED_BYTES = 10 * 1024 * 1024
SECRET_PATTERNS = (
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"(?i)(secret_access_key|password)\s*=\s*[^\s$][^\s]+"),
)


def repository_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def test_no_tracked_large_files_or_sensitive_artifacts() -> None:
    violations: list[str] = []
    for path in repository_files():
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.stat().st_size > MAX_TRACKED_BYTES:
            violations.append(f"large file: {relative}")
        if relative.name == ".env" or relative.parts[0] in {"data", "models"}:
            violations.append(f"sensitive artifact: {relative}")
        if path.stat().st_size < 1024 * 1024 and relative.name != ".env.example":
            content = path.read_bytes()
            for pattern in SECRET_PATTERNS:
                if pattern.search(content):
                    violations.append(f"possible secret: {relative}")
    assert not violations, "\n".join(violations)
