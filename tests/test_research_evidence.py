import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from so101_lab.research_evidence import collection, latency, trials


class ResearchEvidenceTests(unittest.TestCase):
    def test_collection_counts_episodes_and_frames_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "trajectory.bin").write_bytes(b"synthetic fixture")
            digest = hashlib.sha256(b"synthetic fixture").hexdigest()
            row = dict(episode_id="sim-0", source="simulation", robot="fixed-palms", task="test", operator_role="test",
                started_at="2026-10-19T00:00:00Z", ended_at="2026-10-19T00:00:01Z", frames=174, accepted=True,
                artifacts={role: dict(path="trajectory.bin", sha256=digest) for role in ("trajectory", "calibration")})
            report = collection([row], "simulation", root)
            self.assertEqual(report["accepted_episodes"], 1)
            self.assertEqual(report["frames"], 174)
            with self.assertRaises(ValueError):
                collection([row], "real_robot", root)
            with self.assertRaises(ValueError):
                collection([row], "simulation", root, expected_accepted=174)
            broken = copy.deepcopy(row)
            broken["artifacts"]["trajectory"]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                collection([broken], "simulation", root)

    def test_missing_records_and_unreviewed_acceptance_fail(self):
        with self.assertRaises(ValueError):
            collection([], "real_robot", Path.cwd())

    def trial(self, policy, success):
        return dict(source="synthetic", policy=policy, checkpoint="synthetic-fixture", task="test", condition="fixed",
            seed=0, trial_id="0", success=success, failure_type=None if success else "grasp", scene_id="s0", repeat=0,
            environment_seed=123, checkpoint_sha256="a" * 64, protocol_sha256="b" * 64, initial_condition_sha256="c" * 64)

    def test_trials_require_matching_initial_conditions_and_all_policies(self):
        rows = [self.trial("original", False), self.trial("failure_aware", True)]
        self.assertEqual(trials(rows, "synthetic")["paired_conditions"], 1)
        changed = copy.deepcopy(rows)
        changed[1]["environment_seed"] += 1
        with self.assertRaises(ValueError):
            trials(changed, "synthetic")
        changed = copy.deepcopy(rows)
        changed[1]["scene_id"] = "s1"
        with self.assertRaises(ValueError):
            trials(changed, "synthetic")

    def tick(self):
        return dict(source="synthetic", mode="sync", trial_id="t", step=0, clock_id="test-monotonic",
            control_start_ns=0, control_end_ns=200_000_000, blocked_ns=175_000_000,
            capture_ns=0, preprocess_end_ns=10_000_000, inference_start_ns=10_000_000,
            inference_end_ns=175_000_000, send_ns=180_000_000, execute_ns=190_000_000)

    def test_recovery_rates_do_not_pool_training_seeds(self):
        rows = [dict(self.trial(policy, True), seed=seed, recognized_failure=True, autonomous_recovery=(seed == 0))
                for seed in (0, 1) for policy in ("original", "failure_aware")]
        report = trials(rows, "synthetic")
        self.assertEqual(len(report["recovery"]), 4)
        for group in report["recovery"]:
            self.assertEqual(group["recognized_failures"], 1)
            self.assertEqual(group["recovery_rate"], 1. if group["seed"] == "0" else 0.)

    def test_latency_distinguishes_blocking_action_age_and_deadline(self):
        report = latency([self.tick()], "synthetic")["groups"][0]
        self.assertEqual(report["blocked_ms"]["mean"], 175.)
        self.assertEqual(report["action_age_ms"]["mean"], 190.)
        self.assertEqual(report["deadline_misses"], 1)
        self.assertIsNone(report["episode_duration_s"])

    def test_latency_rejects_duplicates_bad_clocks_and_impossible_intervals(self):
        row = self.tick()
        with self.assertRaises(ValueError):
            latency([row, row], "synthetic")
        for key, value in (("inference_end_ns", 1), ("blocked_ns", 999_000_000), ("control_end_ns", -1)):
            changed = dict(row, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                latency([changed], "synthetic")


if __name__ == "__main__":
    unittest.main()
