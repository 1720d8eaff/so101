import importlib.util
from pathlib import Path
import tempfile
from threading import Event
import unittest
import numpy as np
from so101_lab.smolvla_live import SensorSnapshot, SmolVLAStream, async_runner
from so101_lab.runtime import StateAlignedBuffer


class LiveBridgeTests(unittest.TestCase):
    def test_snapshot_owns_camera_and_joint_data(self):
        image = np.zeros((3, 8, 8))
        q = np.zeros(2)
        snapshot = SensorSnapshot(0., q, {"image":image})
        image[:] = 1; q[:] = 1
        self.assertFalse(snapshot.state.any())
        self.assertFalse(snapshot.batch["image"].any())

    def test_async_preserves_image_snapshot(self):
        seen, finished = [], Event()
        def predict(snapshot):
            seen.append(snapshot.batch["camera"].copy())
            finished.set()
            return np.zeros((4, 2))
        snapshot = SensorSnapshot(0., np.zeros(2), {"camera":np.ones((2, 2, 3))})
        buffer = StateAlignedBuffer([0, 0], [-1,-1], [1,1], [.1,.1])
        with async_runner(predict, buffer, sample_dt=.1, clock=lambda:0.) as runner:
            runner.advance(snapshot, now=0.)
            self.assertTrue(finished.wait(2.))
        self.assertEqual(float(seen[0].sum()), 12.)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch")
    def test_processors_are_applied_to_every_action(self):
        import torch
        class Policy:
            def reset(self): pass
            def predict_action_chunk(self, batch):
                self.input = batch
                return torch.ones((1, 3, 2))
        policy = Policy()
        calls = []
        def post(a):
            calls.append(tuple(a.shape))
            return a*2
        stream = SmolVLAStream(policy, lambda batch: {"observation.state": batch["observation.state"]+10}, post)
        snapshot = SensorSnapshot(0., np.zeros(2), {"observation.state": torch.zeros(1,2)})
        np.testing.assert_allclose(stream(snapshot), 2.)
        self.assertEqual(calls, [(1,2)]*3)
        self.assertEqual(float(policy.input["observation.state"].sum()), 20.)

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "optional MuJoCo backend")
    def test_actual_mujoco_fk_and_ik_roundtrip(self):
        from so101_lab.mujoco_backend import MujocoKinematics
        xml = '<mujoco><worldbody><body><joint name="x" type="slide" axis="1 0 0" range="-1 1" limited="true"/><geom size=".02" mass="1"/><site name="tip"/></body></worldbody></mujoco>'
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"robot.xml"; path.write_text(xml)
            fk = MujocoKinematics(path, ["x"], eef_site="tip")
            self.assertAlmostEqual(fk([.3])[0,3], .3)
            adapter = fk.adapter()
            action = adapter.encode([0.], [.3], .5)
            joints, gripper = adapter.decode([0.], action)
            np.testing.assert_allclose(joints, [.3], atol=1e-4)
            self.assertEqual(gripper, .5)


if __name__ == "__main__":
    unittest.main()
