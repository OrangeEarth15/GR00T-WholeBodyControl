"""Exercise the real motion loader, including OmegaConf and legacy sampling."""

import json
from pathlib import Path
import random
import tempfile
import unittest

from omegaconf import OmegaConf

from gear_sonic.utils.motion_lib.motion_lib_base import MotionLibBase


class MotionKeySelectionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for key in ("walk[0]", "motion.*", "stand", "walk${seed}", "walk${missing}", "walk0"):
            (self.root / f"{key}.pkl").touch()
        self.keys_file = self.root / "keys.json"

    def select(self, keys, **overrides):
        cfg = {"seed": 0, "motion_keys_file": str(self.keys_file), **overrides}
        self.keys_file.write_text(json.dumps(keys), encoding="utf-8")
        lib = object.__new__(MotionLibBase)
        lib.debug = False
        lib.m_cfg = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
        lib.load_data(str(self.root))
        return lib._motion_data_keys.tolist()

    def test_exact_order_overrides_random_cap(self):
        keys = ["stand", "walk[0]", "motion.*"]
        self.assertEqual(self.select(keys, max_unique_motions=1), keys)

    def test_config_interpolation_does_not_rewrite_keys(self):
        keys = ["walk${seed}", "walk0", "walk${missing}"]
        self.assertEqual(self.select(keys, max_unique_motions=1), keys)

    def test_missing_and_filtered_keys_fail(self):
        for keys, cfg in ((["walk0-missing"], {}), (["stand"], {"remove_motion_keys": ["stand"]}),
                          (["stand"], {"filter_motion_keys": ["walk[0]"]})):
            with self.subTest(cfg=cfg), self.assertRaisesRegex(ValueError, "unavailable after filtering"):
                self.select(keys, **cfg)

    def test_default_sampling_still_uses_seed_and_cap(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(13)
        first = self.select(None, motion_keys_file=None, max_unique_motions=2)
        random.seed(13)
        second = self.select(None, motion_keys_file=None, max_unique_motions=2)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)


if __name__ == "__main__":
    unittest.main()
