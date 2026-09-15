from types import SimpleNamespace

import pytest
import torch

from lerobot.scripts.valid_dataset import validate_on_dataset


class _Policy:
    def __init__(self):
        self.seen_batches = []

    def eval(self):
        return self

    def predict_action_chunk(self, batch):
        self.seen_batches.append(batch)
        assert "action" not in batch
        return torch.tensor([[[1.0, 3.0], [99.0, 99.0]]])

    def select_action(self, batch):
        raise AssertionError("offline validation must not use stateful online inference")


def _cfg(save_predictions=False):
    return SimpleNamespace(
        batch_size=1,
        num_workers=0,
        policy=SimpleNamespace(use_amp=False),
        action_success_threshold=1.0,
        save_predictions=save_predictions,
    )


def test_validation_uses_first_action_from_stateless_chunk_prediction():
    dataset = [{"observation.state": torch.zeros(2), "action": torch.tensor([[1.0, 3.0], [5.0, 5.0]])}]
    results = validate_on_dataset(dataset, _Policy(), _cfg(save_predictions=True), torch.device("cpu"))

    assert results["aggregated"]["total_samples"] == 1
    assert results["aggregated"]["avg_action_rmse"] == pytest.approx(0.0)
    assert results["predictions"]["predicted_actions"].shape == (1, 2)


def test_validation_rejects_empty_dataset():
    with pytest.raises(ValueError, match="empty dataset"):
        validate_on_dataset([], _Policy(), _cfg(), torch.device("cpu"))
