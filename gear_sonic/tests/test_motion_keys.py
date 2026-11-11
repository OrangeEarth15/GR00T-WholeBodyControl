"""Validate explicit evaluation sets before starting a simulator."""

import json
from pathlib import Path
import tempfile
import unittest

from gear_sonic.utils.motion_keys import load_motion_keys


class MotionKeysTest(unittest.TestCase):
    def load(self, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "keys.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            return load_motion_keys(path)

    def test_order_and_literal_names(self):
        keys = ["walk[0]", "motion.*", "walk${seed}", "walk${missing}", "stand"]
        self.assertEqual(self.load(keys), keys)

    def test_reject_duplicate_keys(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            self.load(["walk", "walk"])

    def test_reject_empty_or_wrong_container(self):
        for value in ([], {}, "walk", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonempty JSON list"):
                self.load(value)

    def test_reject_invalid_keys(self):
        for value in ([""], [" "], [1], [None], [["walk"]]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonempty string"):
                self.load(value)


if __name__ == "__main__":
    unittest.main()
