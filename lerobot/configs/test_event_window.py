import unittest

from lerobot.configs import parser


class EventWindowCliTest(unittest.TestCase):
    def test_requires_exactly_one_mode(self):
        with self.assertRaisesRegex(ValueError, "neither was provided"):
            parser.resolve_event_window_args("policy", [])
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            parser.resolve_event_window_args(
                "policy", ["--policy.event_window_size=2000", "--policy.event_window_duration=20000"]
            )

    def test_accepts_one_mode_across_compatible_namespaces(self):
        self.assertEqual(
            parser.resolve_event_window_args(
                ("dataset", "policy"),
                ["--dataset.event_window_size=2000", "--policy.event_window_size=2000"],
            ),
            (2000, None),
        )

    def test_rejects_conflicting_repeated_values(self):
        with self.assertRaisesRegex(ValueError, "Conflicting runtime values"):
            parser.resolve_event_window_args(
                ("dataset", "policy"),
                ["--dataset.event_window_size=1000", "--policy.event_window_size=2000"],
            )

    def test_training_selection_clears_checkpoint_counterpart(self):
        overrides = parser.set_event_window_cli_overrides(["--device=cpu"], None, 20000)
        self.assertEqual(
            overrides,
            ["--device=cpu", "--event_window_size=null", "--event_window_duration=20000"],
        )

    def test_inference_ignores_window_overrides(self):
        self.assertEqual(
            parser.strip_event_window_cli_overrides(
                ["--device=cpu", "--event_window_size=2000", "--event_window_duration=null"]
            ),
            ["--device=cpu"],
        )


if __name__ == "__main__":
    unittest.main()
