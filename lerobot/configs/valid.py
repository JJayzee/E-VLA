import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig


@dataclass
class ValidatePipelineConfig:
    """Configuration for the validation pipeline."""
    dataset: DatasetConfig
    policy: PreTrainedConfig | None = None
    output_dir: Path | None = None
    job_name: str | None = None
    seed: int | None = 1000
    num_workers: int = 4
    batch_size: int = 16
    action_success_threshold: float = 2.0   # diagnostic RMSE cutoff, not task success
    save_predictions: bool = False          # True to save all predictions to a txt file

    def __post_init__(self):
        # Parse policy path from CLI arguments
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.strip_event_window_cli_overrides(parser.get_cli_overrides("policy"))
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            raise ValueError("A policy path must be provided for validation")

        # Offline event samples still need a dataset-side window. Read it from
        # the checkpoint-loaded policy; inference CLI arguments do not override it.
        self.dataset.event_window_size = self.policy.event_window_size
        self.dataset.event_window_duration = self.policy.event_window_duration

        if not self.job_name:
            self.job_name = f"valid_{self.policy.type}"

        if not self.output_dir:
            now = dt.datetime.now()
            valid_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/valid") / valid_dir

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
