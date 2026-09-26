#!/usr/bin/env python3
"""Apply the small upstream contribution overlay to the exact supported revision."""
import argparse
from pathlib import Path
import shutil
import subprocess

REVISION = "1760db148b951363f2282261f30179dfd2ce3790"


def install(root):
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise SystemExit(f"unsupported VectorDBBench revision {revision}; expected {REVISION}")
    clients = root / "vectordb_bench/backend/clients"
    registry = clients / "__init__.py"
    cli = root / "vectordb_bench/cli/vectordbbench.py"
    edits = [('    Milvus = "Milvus"', '    KTANN = "KTANN"\n    Milvus = "Milvus"'),
             ('        if self == DB.Milvus:\n            from .milvus.milvus import Milvus',
              '        if self == DB.KTANN:\n            from .ktann.ktann import KTANN\n\n            return KTANN\n\n        if self == DB.Milvus:\n            from .milvus.milvus import Milvus'),
             ('        if self == DB.Milvus:\n            from .milvus.config import MilvusConfig',
              '        if self == DB.KTANN:\n            from .ktann.config import KTANNConfig\n\n            return KTANNConfig\n\n        if self == DB.Milvus:\n            from .milvus.config import MilvusConfig'),
             ('        if self == DB.Milvus:\n            from .milvus.config import _milvus_case_config',
              '        if self == DB.KTANN:\n            from .ktann.config import KTANNCaseConfig\n\n            return KTANNCaseConfig\n\n        if self == DB.Milvus:\n            from .milvus.config import _milvus_case_config')]
    source = registry.read_text()
    original = source
    for old, new in edits:
        if new not in source:
            if source.count(old) != 1:
                raise SystemExit(f"registry anchor mismatch: {old}")
            source = source.replace(old, new)
    cli_original = cli.read_text()
    cli_source = cli_original
    if 'from ..backend.clients.ktann.cli import KTANN' not in cli_source:
        cli_source = cli_source.replace('from ..backend.clients.zvec.cli import Zvec', 'from ..backend.clients.ktann.cli import KTANN\nfrom ..backend.clients.zvec.cli import Zvec')
        cli_source = cli_source.replace('cli.add_command(Zvec)', 'cli.add_command(KTANN)\ncli.add_command(Zvec)')
    if source != original:
        registry.write_text(source)
    if cli_source != cli_original:
        cli.write_text(cli_source)
    shutil.copytree(Path(__file__).parent / "ktann", clients / "ktann", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    install(parser.parse_args().checkout.resolve())
