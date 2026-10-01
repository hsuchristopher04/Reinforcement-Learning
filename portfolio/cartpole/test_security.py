"""Regression checks for artifact boundaries and malformed tree traversal."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import distill
import project


class SecurityChecks(unittest.TestCase):
    def test_manifest_cannot_escape_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'artifacts').mkdir()
            (root / 'artifacts/manifest.json').write_text(json.dumps({
                'files': [{'path': '../outside', 'bytes': 0, 'sha256': ''}]}))
            with patch.object(project, 'ROOT', root), self.assertRaises(SystemExit):
                project.verify()

    def test_cyclic_tree_fails_instead_of_hanging(self):
        model = dict(children_left=[0], children_right=[0], feature=[0], threshold=[0], node_action=[0])
        with self.assertRaisesRegex(ValueError, 'cyclic'):
            distill.tree_action(model, [0, 0, 0, 0])

    def test_negative_child_is_rejected(self):
        model = dict(children_left=[-2], children_right=[0], feature=[0], threshold=[0], node_action=[0])
        with self.assertRaisesRegex(ValueError, 'bounds'):
            distill.tree_action(model, [0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
