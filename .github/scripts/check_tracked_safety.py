"""Reject local-only files and common credential patterns without printing values.

Reads the Git index so the same command checks staged content before commit and
the checked-out commit in CI. This is a small guard, not an exhaustive scanner.
"""

from pathlib import PurePosixPath
import re
import subprocess


TOKEN_PATTERNS = (
    rb"\bsk-(?:or-v1-)?[A-Za-z0-9_-]{32,}",
    rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})",
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
)
LOCAL_DIRS = {"logs", "results", ".venv", "venv", "env", "secrets", "private", "confidential"}


def forbidden_path(path):
    p = PurePosixPath(path)
    name = p.name.lower()
    if any(part.lower() in LOCAL_DIRS for part in p.parts[:-1]):
        return True
    if name == ".env.example":
        return False
    return (
        name == ".env" or name.startswith(".env.")
        or name.endswith(".env") or ".env." in name
        or p.suffix.lower() in {".key", ".pem", ".p12", ".pfx", ".log"}
        or (name.startswith("credentials") and name.endswith(".json"))
        or ("service-account" in name and name.endswith(".json"))
    )


def main():
    paths = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    failures = []
    for path in filter(None, paths):
        if forbidden_path(path):
            failures.append((path, "local-only path"))
            continue
        content = subprocess.check_output(["git", "show", ":" + path])
        if any(re.search(pattern, content) for pattern in TOKEN_PATTERNS):
            failures.append((path, "credential pattern"))
        if PurePosixPath(path).name.lower() == ".env.example":
            for line in content.decode("utf-8-sig").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    value = line.split("=", 1)[1].strip().strip("\"").strip("'")
                    if value:
                        failures.append((path, "example values must be empty"))
                        break
    for path, rule in failures:
        print(f"FAIL: {path}: {rule}")
    if not failures:
        print("Tracked-file safety check passed; review new files and the staged diff separately.")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
