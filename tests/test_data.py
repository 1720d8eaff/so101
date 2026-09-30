import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from so101_lab.data import fit_statistics, select_rows, split_episodes, validate_manifest


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def episode(self, name, value):
        np.savez(self.root / f"{name}.npz", observation_state=np.full((3, 2), value), action=np.full((3, 2), value))
        return {"episode_id": name, "npz_path": f"{name}.npz", "success": True, "timeout": False, "collision": False, "abnormal_action": False}

    def manifest(self, rows):
        p = self.root / "episodes.jsonl"
        p.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
        return p

    def test_validation_rejects_nonfinite_and_misaligned(self):
        rows = [self.episode("good", 1), self.episode("bad", np.nan), self.episode("misaligned", 1)]
        np.savez(self.root / "misaligned.npz", observation_state=np.ones((3, 2)), action=np.ones((2, 2)))
        _, checks = validate_manifest(self.manifest(rows))
        self.assertEqual([x["valid"] for x in checks], [True, False, False])

    def test_unknown_flags_do_not_silently_pass(self):
        rows = [self.episode("good", 1), self.episode("unknown", 2)]
        rows[1].pop("collision")
        values, checks = validate_manifest(self.manifest(rows))
        selected, rejected = select_rows(values, checks)
        self.assertEqual([x["episode_id"] for x in selected], ["good"])
        self.assertIn("collision_true_or_unknown", rejected[0]["reasons"])

    def test_split_is_deterministic_disjoint_and_order_independent(self):
        rows = [{"episode_id": str(x)} for x in range(10)]
        first = split_episodes(rows, 0.8, 42)
        self.assertEqual(first, split_episodes(list(reversed(rows)), 0.8, 42))
        self.assertEqual(len(first["train"]), 8)
        self.assertFalse(set(first["train"]) & set(first["validation"]))

    def test_statistics_exclude_validation(self):
        rows = [self.episode("train-a", 1), self.episode("train-b", 3), self.episode("validation", 100000)]
        result = fit_statistics(self.manifest(rows), rows, {"train": ["train-a", "train-b"], "validation": ["validation"]})
        self.assertEqual(result["statistics"]["action"]["mean"], [2, 2])
        self.assertEqual(result["statistics"]["action"]["std"], [1, 1])
        self.assertEqual(result["statistics"]["action"]["frames"], 6)

    def test_statistics_reject_overlap_and_missing_ids(self):
        rows = [self.episode("a", 1), self.episode("b", 2)]
        p = self.manifest(rows)
        for split in ({"train": ["a"], "validation": ["a", "b"]}, {"train": ["a"], "validation": []}):
            with self.assertRaises(ValueError):
                fit_statistics(p, rows, split)

    def test_duplicate_episode_is_rejected(self):
        row = self.episode("a", 1)
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest([row, row]))
