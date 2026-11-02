"""Skip Rust jobs only for changes to known, non-compiled documentation."""

import subprocess
import sys
from pathlib import PurePosixPath


def documentation_only(path):
    """Unknown paths and the Memory README's compiled examples need Rust CI."""
    return path in {
        "README.md",
        "CONTEXT.md",
        "AGENTS.md",
        "benchmarks/README.md",
        "ktann-rocksdb/README.md",
        "ktann-foundationdb/README.md",
    } or (path.startswith("docs/") and PurePosixPath(path).suffix == ".md")


if __name__ == "__main__":
    base, head = sys.argv[1:]
    # A first push has no previous commit: run the full suite.
    code = True
    if base and set(base) != {"0"}:
        paths = subprocess.check_output(
            ["git", "diff", "--no-renames", "--name-only", "-z", base, head]
        ).decode().split("\0")
        code = any(not documentation_only(path) for path in paths if path)
    print(f"code={str(code).lower()}")
