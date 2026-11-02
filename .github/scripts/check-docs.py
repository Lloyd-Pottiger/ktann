"""Check local inline Markdown links in the tracked architecture documentation.

External URLs are deliberately excluded: network availability is not a document
correctness check. Fenced examples are excluded from link and heading discovery.
"""

import re
import subprocess
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit


def prose(path):
    """Return Markdown outside fenced code blocks."""
    lines = []
    fence = None
    for line in path.read_text().splitlines():
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
        elif fence is None:
            lines.append(line)
    return "\n".join(lines)


def anchors(path):
    """Build GitHub-style ATX heading anchors, including duplicate suffixes."""
    result = set()
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*$", prose(path), re.MULTILINE):
        heading = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", heading)
        slug = "".join(
            char for char in heading.lower().strip()
            if char in "_- " or unicodedata.category(char)[0] in "LN"
        ).replace(" ", "-")
        anchor = slug
        number = 0
        while anchor in result:
            number += 1
            anchor = f"{slug}-{number}"
        result.add(anchor)
    return result


def destinations(markdown):
    """Extract inline destinations without angle wrappers or optional titles."""
    pattern = r"""\[[^\]]*\]\(\s*(?:<([^>\n]*)>|((?:[^\s()]|\([^()]*\))*))(?:\s+(?:"[^"\n]*"|'[^'\n]*'|\([^()\n]*\)))?\s*\)"""
    return [angle or plain for angle, plain in re.findall(pattern, markdown)]


def check():
    """Check tracked docs only; ignored research notes are not CI inputs."""
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "docs", "README.md", "CONTEXT.md", "AGENTS.md",
         "benchmarks/README.md", "ktann-rocksdb/README.md", "ktann-foundationdb/README.md"]
    ).decode().split("\0")
    failures = []
    count = 0
    for name in paths:
        path = Path(name)
        if not name or path.suffix != ".md":
            continue
        for target in destinations(prose(path)):
            url = urlsplit(target)
            if url.scheme or url.netloc:
                continue
            count += 1
            linked = path.parent / unquote(url.path) if url.path else path
            if not linked.exists():
                failures.append(f"{path}: missing target {target}")
            elif url.fragment and linked.suffix == ".md":
                if unquote(url.fragment) not in anchors(linked):
                    failures.append(f"{path}: missing heading {target}")
    print(f"Checked {count} local documentation links.")
    for failure in failures:
        print(failure)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(check())
