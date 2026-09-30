import json
import unittest
from pathlib import Path
from unittest.mock import patch
from so101_lab.launch import ACTIONS, build_command, main
from so101_lab.report import summarize

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "so101.example.json"


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_every_action_has_expected_entrypoint(self):
        for action in ACTIONS:
            cmd = build_command(action, self.cfg)
            self.assertTrue(cmd[0].startswith("lerobot-"))
            self.assertTrue(all(isinstance(x, str) for x in cmd))

    def test_preview_does_not_execute_subprocess(self):
        with patch("so101_lab.launch.subprocess.run") as run, patch("builtins.print"):
            self.assertEqual(main(["record", "--config", str(CONFIG)]), 0)
            run.assert_not_called()

    def test_uploads_disabled(self):
        for action in ("record", "evaluate-robot"):
            self.assertIn("--dataset.push_to_hub=false", build_command(action, self.cfg))
        for action in ("train-smolvla", "train-act", "train-diffusion"):
            self.assertIn("--policy.push_to_hub=false", build_command(action, self.cfg))

    def test_evaluation_uses_new_rollout_api(self):
        cmd = build_command("evaluate-robot", self.cfg)
        self.assertEqual(cmd[0], "lerobot-rollout")
        self.assertIn("--strategy.type=episodic", cmd)
        self.assertIn("--device=cuda", cmd)
        self.assertNotIn("--policy.device=cuda", cmd)

    def test_pretrained_diffusion_initialization(self):
        self.cfg["training"]["init_policies"]["diffusion"] = "checkpoints/sim/pretrained_model"
        cmd = build_command("train-diffusion", self.cfg)
        self.assertIn("--policy.path=checkpoints/sim/pretrained_model", cmd)
        self.assertNotIn("--policy.type=diffusion", cmd)

    def test_episode_indices_forwarded_and_invalid_indices_rejected(self):
        self.cfg["training"]["episodes"] = [0, 2, 4]
        self.assertIn("--dataset.episodes=[0,2,4]", build_command("train-act", self.cfg))
        self.cfg["training"]["episodes"] = [0, 0]
        with self.assertRaises(ValueError):
            build_command("train-act", self.cfg)


class ReportTests(unittest.TestCase):
    @staticmethod
    def trial(identifier, success, seed=0):
        return dict(policy="smolvla", checkpoint="run-a", task="pick-place", condition="fixed", seed=seed,
                    trial_id=identifier, success=success, failure_type=None if success else "grasp")

    def test_groups_seeds_separately_and_counts_failures(self):
        result = summarize([self.trial("a", True), self.trial("b", False), self.trial("c", True, 1)])
        self.assertEqual(len(result["groups"]), 2)
        self.assertEqual(result["groups"][0]["success_rate"], 0.5)
        self.assertEqual(result["groups"][0]["failures"], {"grasp": 1})

    def test_duplicate_and_unlabeled_trials_rejected(self):
        row = self.trial("a", True)
        with self.assertRaises(ValueError):
            summarize([row, row])
        row["success"] = "unknown"
        with self.assertRaises(ValueError):
            summarize([row])
