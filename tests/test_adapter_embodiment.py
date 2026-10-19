import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from so101_lab.embodiment import (IKConvergenceError, KinematicsAdapter, check_pose, decode_delta,
                                 encode_delta, exp_rotation, robot_task_holdout)
try:
    import torch
    from so101_lab.adapter import (AdapterConfig, AdaptedChunkPolicy, HistoryFailureAdapter,
                                   adapter_loss, load_adapter, save_adapter)
except ImportError:
    torch = None


class EmbodimentTests(unittest.TestCase):
    def fk(self, q):
        t = np.eye(4)
        t[:3, 3] = q
        return t

    def test_body_frame_and_absolute_gripper_round_trip(self):
        current = np.eye(4)
        current[:3, :3] = exp_rotation([0., 0., np.pi / 2])
        target = current.copy()
        target[:3, 3] += [.1, 0., 0.]
        action = encode_delta(current, target, .7)
        np.testing.assert_allclose(action[:3], [0., -.1, 0.], atol=1e-8)
        decoded, grip = decode_delta(current, action)
        np.testing.assert_allclose(decoded, target, atol=1e-8)
        self.assertEqual(grip, .7)

    def test_rotation_near_pi_and_reflections(self):
        current, target = np.eye(4), np.eye(4)
        target[:3, :3] = exp_rotation([np.pi, 0., 0.])
        decoded, _ = decode_delta(current, encode_delta(current, target, 0.))
        np.testing.assert_allclose(decoded, target, atol=1e-7)
        target[0, 0] = -1
        with self.assertRaises(ValueError):
            check_pose(target)

    def test_bounded_ik_round_trip_and_unreachable_target(self):
        adapter = KinematicsAdapter(self.fk, [-.2] * 3, [.2] * 3)
        action = adapter.encode([0, 0, 0], [.05, -.03, .02], .8)
        q, gripper = adapter.decode([0, 0, 0], action)
        np.testing.assert_allclose(q, [.05, -.03, .02], atol=1e-4)
        self.assertEqual(gripper, .8)
        target = self.fk([1, 0, 0])
        with self.assertRaises(IKConvergenceError):
            adapter.solve(target, [0, 0, 0])

    def test_unseen_robot_task_split_keeps_whole_episodes(self):
        episodes = [dict(episode_id="a", robot_id="panda", task="pick"),
                    dict(episode_id="b", robot_id="ur5e", task="place"),
                    dict(episode_id="c", robot_id="panda", task="place")]
        split = robot_task_holdout(episodes, [("panda", "place")])
        self.assertEqual(split["validation"], ["c"])
        self.assertEqual(split["train"], ["a", "b"])
        with self.assertRaises(ValueError):
            robot_task_holdout(episodes, [("so101", "place")])

    def test_ik_handles_rotation_and_translation_together(self):
        def fk(q):
            t = np.eye(4)
            t[:3, 3], t[:3, :3] = q[:3], exp_rotation(q[3:])
            return t
        adapter = KinematicsAdapter(fk, [-1.] * 6, [1.] * 6)
        desired = [.03, -.02, .01, .12, -.08, .04]
        solved = adapter.solve(fk(desired), [0.] * 6)
        np.testing.assert_allclose(fk(solved), fk(desired), atol=1e-4)


@unittest.skipIf(torch is None, "Install the learning extra to test adapters")
class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        self.model = HistoryFailureAdapter(AdapterConfig(3, 2, history=4, horizon=3))
        self.history = torch.randn(6, 4, 3)
        self.mask = torch.ones(6, 4, dtype=torch.bool)
        self.base = torch.zeros(6, 3, 2)

    def test_zero_init_preserves_base_and_training_updates_model(self):
        output = self.model(self.history, self.mask, self.base)
        torch.testing.assert_close(output["actions"], self.base, rtol=0, atol=0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.01)
        action_mask = torch.ones(6, 3, dtype=torch.bool)
        target = torch.full_like(self.base, .2)
        before = float(adapter_loss(output, target, action_mask)["action"].detach())
        for _ in range(40):
            optimizer.zero_grad()
            output = self.model(self.history, self.mask, self.base)
            losses = adapter_loss(output, target, action_mask, torch.ones(6, dtype=torch.long), torch.ones(6) * .5)
            losses["total"].backward()
            optimizer.step()
        after = float(adapter_loss(self.model(self.history, self.mask, self.base), target, action_mask)["action"].detach())
        self.assertLess(after, before / 10)

    def test_padding_cannot_change_prediction_or_loss(self):
        self.mask[:, 0] = False
        with torch.no_grad():
            self.model.residual[-1].weight.fill_(.01)
        alternate = self.history.clone()
        alternate[:, 0] = 999
        torch.testing.assert_close(self.model(self.history, self.mask, self.base)["actions"],
                                   self.model(alternate, self.mask, self.base)["actions"])
        output = self.model(self.history, self.mask, self.base)
        action_mask = torch.ones(6, 3, dtype=torch.bool)
        action_mask[:, -1] = False
        target = torch.zeros_like(self.base)
        changed = target.clone()
        changed[:, -1] = 999
        torch.testing.assert_close(adapter_loss(output, target, action_mask)["action"],
                                   adapter_loss(output, changed, action_mask)["action"])

    def test_unknown_labels_do_not_create_nan(self):
        output = self.model(self.history, self.mask, self.base)
        losses = adapter_loss(output, self.base, torch.ones(6, 3, dtype=torch.bool),
                              torch.full((6,), -1, dtype=torch.long), torch.full((6,), -1.))
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(float(losses["failure"].detach()), 0.)
        self.assertEqual(float(losses["progress"].detach()), 0.)

    def test_checkpoint_round_trip_and_refusal_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "adapter.pt"
            save_adapter(path, self.model, {"source": "synthetic_test"})
            loaded, metadata = load_adapter(path)
            self.assertEqual(metadata["source"], "synthetic_test")
            torch.testing.assert_close(self.model(self.history, self.mask, self.base)["actions"],
                                       loaded(self.history, self.mask, self.base)["actions"])
            with self.assertRaises(ValueError):
                save_adapter(path, self.model, {})

    def test_wrapper_resets_history_and_calls_chunk_api(self):
        class Base:
            resets = 0
            def reset(self):
                self.resets += 1
            def predict_action_chunk(self, batch):
                return torch.zeros(1, 4, 2)
        base = Base()
        wrapper = AdaptedChunkPolicy(base, self.model)
        self.assertEqual(tuple(wrapper.predict_action_chunk({"observation.state": torch.ones(1, 3)}).shape), (1, 3, 2))
        self.assertEqual(len(wrapper.history), 1)
        wrapper.reset()
        self.assertEqual(len(wrapper.history), 0)
        self.assertEqual(base.resets, 1)

    def test_training_uses_only_training_episodes_and_checks_normalization_ids(self):
        from so101_lab.demo import synthetic_cache
        from so101_lab.train_adapter import train
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            cache, split, metadata = synthetic_cache(directory / "data")
            result = train(cache, split, metadata, directory / "trained", epochs=3)
            self.assertEqual(result["train_examples"], 48)
            self.assertEqual(result["validation_examples"], 16)
            provenance = json.loads((directory / "trained/run.json").read_text())
            self.assertFalse(set(provenance["train_episodes"]) & set(provenance["validation_episodes"]))
            bad = json.loads(metadata.read_text())
            bad["normalization_training_episodes"].append("ep-3")
            metadata.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "normalization"):
                train(cache, split, metadata, directory / "bad", epochs=1)
            self.assertFalse((directory / "bad").exists())


if __name__ == "__main__":
    unittest.main()
