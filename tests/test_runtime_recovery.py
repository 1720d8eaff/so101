import copy
import threading
import unittest
import numpy as np
from so101_lab.recovery import (FAILURE_CLASSES, FailureEvent, action_targets, failure_labels,
                                history_windows, recovery_windows, training_weights)
from so101_lab.runtime import ActionChunk, AsyncInference, BufferSettings, Observation, StateAlignedBuffer


class BufferTests(unittest.TestCase):
    def buffer(self, **settings):
        return StateAlignedBuffer([0., 0.], [-1., -1.], [1., 1.], .1,
            BufferSettings(max_pending_steps=8, replan_below_steps=2, **settings))

    def chunk(self, request=0, obs=0., generated=.01, actions=None, state=None):
        return ActionChunk(request, obs, generated, .05, [0., 0.] if state is None else state,
                           np.ones((6, 2)) if actions is None else actions)

    def test_delayed_prefix_is_discarded_and_replacement_is_not_append(self):
        buffer = self.buffer()
        self.assertEqual(buffer.submit(self.chunk(), .11, [0, 0]), "accepted")
        self.assertEqual(buffer.counters["discarded_prefix_steps"], 2)
        self.assertEqual(buffer.tick(.11, .11).status, "hold")
        np.testing.assert_allclose(buffer.tick(.15, .15).action, [.1, .1])
        self.assertEqual(buffer.submit(self.chunk(1, .1, .16), .16, [0, 0]), "accepted")
        self.assertLessEqual(len(buffer.queue), 6)

    def test_rejects_stale_drift_and_reordered_results(self):
        buffer = self.buffer()
        self.assertEqual(buffer.submit(self.chunk(), .01, [0, 0]), "accepted")
        self.assertEqual(buffer.submit(self.chunk(), .02, [0, 0]), "out_of_order")
        self.assertEqual(buffer.submit(self.chunk(1), .6, [0, 0]), "stale_observation")
        self.assertEqual(buffer.submit(self.chunk(2), .1, [1, 0]), "state_drift")

    def test_limits_step_size_and_hold_on_sensor_timeout(self):
        buffer = self.buffer()
        buffer.submit(self.chunk(actions=np.full((6, 2), 100.)), .01, [0, 0])
        previous = np.zeros(2)
        for tick in (.05, .1, .15, .2):
            step = buffer.tick(tick, tick)
            self.assertTrue((np.abs(step.action - previous) <= .10000001).all())
            self.assertTrue((step.action <= 1).all())
            previous = step.action
        step = buffer.tick(.8, .2)
        np.testing.assert_array_equal(step.action, previous)
        self.assertEqual(step.status, "stale_observation")
        self.assertTrue(step.request_replan)

    def test_late_control_tick_never_executes_old_action(self):
        buffer = self.buffer(max_action_lag_s=.01)
        buffer.submit(self.chunk(), .01, [0, 0])
        step = buffer.tick(.49, .49)
        self.assertEqual(step.status, "hold")
        np.testing.assert_array_equal(step.action, [0., 0.])

    def test_future_targets_with_stale_anchor_do_not_delay_replanning(self):
        buffer = self.buffer()
        chunk = ActionChunk(0, 0., .01, .2, [0, 0], np.ones((8, 2)))
        buffer.submit(chunk, .01, [0, 0])
        step = buffer.tick(.51, .51)
        self.assertEqual(step.pending_steps, 0)
        self.assertTrue(step.request_replan)
        self.assertEqual(step.status, "hold")

    def test_invalid_values_future_clocks_and_empty_chunks(self):
        for actions in (np.empty((0, 2)), [[float("nan"), 0]]):
            with self.assertRaises(ValueError):
                self.chunk(actions=actions)
        with self.assertRaises(ValueError):
            self.buffer().submit(self.chunk(generated=1.), .5, [0, 0])
        buffer = self.buffer()
        buffer.tick(.1, .1)
        with self.assertRaises(ValueError):
            buffer.tick(.09, .09)

    def test_snapshot_copies_mutable_inputs(self):
        actions, state = np.ones((3, 2)), np.zeros(2)
        chunk = self.chunk(actions=actions, state=state)
        actions[:] = 99
        state[:] = 99
        np.testing.assert_array_equal(chunk.actions, np.ones((3, 2)))
        np.testing.assert_array_equal(chunk.reference_state, [0, 0])

    def test_slow_inference_does_not_block_control_or_queue_more_work(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        clock_value = [.01]
        def predict(observation):
            calls.append(observation)
            entered.set()
            if not release.wait(2):
                raise RuntimeError("Test did not release inference")
            return np.ones((6, 2)) * .2
        runner = AsyncInference(predict, self.buffer(), .05, clock=lambda: clock_value[0])
        try:
            runner.advance(Observation(0., [0, 0]), .01)
            self.assertTrue(entered.wait(1))
            first_future = runner.future
            for t in (.02, .03, .04):
                runner.advance(Observation(t, [0, 0]), t)
                self.assertIs(runner.future, first_future)
            self.assertEqual(len(calls), 1)
            clock_value[0] = .04
            release.set()
            first_future.result(timeout=2)
            runner.advance(Observation(.05, [0, 0]), .05)
            self.assertEqual(runner.buffer.counters["accepted"], 1)
        finally:
            release.set()
            runner.close()


class RecoveryTests(unittest.TestCase):
    def test_history_is_causal_and_resets_at_episode_boundary(self):
        states = np.arange(10).reshape(5, 2)
        windows, mask = history_windows(states, 3)
        altered = states.copy()
        altered[3:] = 999
        other, _ = history_windows(altered, 3)
        np.testing.assert_array_equal(windows[:3], other[:3])
        np.testing.assert_array_equal(mask[0], [False, False, True])
        fresh, mask = history_windows([[100., 200.]], 3)
        self.assertEqual(fresh[0, 0, 0], 0)
        np.testing.assert_array_equal(mask[0], [False, False, True])

    def test_action_horizon_does_not_cross_episode(self):
        targets, mask = action_targets([[1.], [2.]], 3)
        np.testing.assert_array_equal(targets[-1, :, 0], [2, 0, 0])
        np.testing.assert_array_equal(mask[-1], [True, False, False])

    def test_unknown_labels_and_merged_recovery_windows(self):
        events = [FailureEvent(3, 5, "grasp", True), FailureEvent(7, 9, "target_change", False)]
        labels = failure_labels(12, events)
        self.assertEqual(labels[0], -1)
        self.assertEqual(labels[3], FAILURE_CLASSES.index("grasp"))
        self.assertEqual(recovery_windows(12, events, 2, 2), [[1, 11]])
        with self.assertRaises(ValueError):
            failure_labels(12, [events[0], events[0]])

    def test_sampler_never_includes_validation(self):
        indices, weights = training_weights(["a", "a", "b", "b"], [-1, 2, 2, 2], {"train": ["a"], "validation": ["b"]})
        np.testing.assert_array_equal(indices, [0, 1])
        np.testing.assert_array_equal(weights, [1., 3.])
        with self.assertRaises(ValueError):
            training_weights(["a", "b"], [0, 1], {"train": ["a", "b"], "validation": ["b"]})


if __name__ == "__main__":
    unittest.main()
