"""Regression checks for CI selection and documentation link handling."""

import importlib.util
import unittest
from pathlib import Path


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scope = load("ci-scope")
docs = load("check-docs")


class CiChecks(unittest.TestCase):
    def test_compiled_and_unknown_paths_require_rust(self):
        for path in ["ktann-memory/README.md", "src/lib.rs", "Cargo.lock",
                     ".github/workflows/ci.yml", "docs/example.rs", "new/README.md"]:
            with self.subTest(path=path):
                self.assertFalse(scope.documentation_only(path))

    def test_known_documentation_skips_rust(self):
        for path in ["README.md", "docs/design/api.md", "docs/adr/new.md"]:
            with self.subTest(path=path):
                self.assertTrue(scope.documentation_only(path))

    def test_link_titles_and_wrappers_are_not_destinations(self):
        cases = {
            '[Examples](#examples "Jump to examples")': "#examples",
            '[Examples](<#examples> "Jump to examples")': "#examples",
            '[File](<file with spaces.md>)': "file with spaces.md",
            "[File](file.md 'title')": "file.md",
            "[File](file.md (title))": "file.md",
            "[File](file(1).md)": "file(1).md",
        }
        for markdown, destination in cases.items():
            with self.subTest(markdown=markdown):
                self.assertEqual(docs.destinations(markdown), [destination])


if __name__ == "__main__":
    unittest.main()
