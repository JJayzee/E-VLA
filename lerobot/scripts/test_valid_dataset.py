from types import SimpleNamespace

import torch

from lerobot.scripts.valid_dataset import compute_action_rmse, validate_on_dataset


class _OfflinePolicy:
    def __init__(self):
        self.chunk_calls = 0

    def eval(self):
        return self

    def predict_action_chunk(self, batch):
        self.chunk_calls += 1
        batch_size = batch["observation.state"].shape[0]
        chunk = torch.zeros((batch_size, 2, 2))
        chunk[:, 0] = batch["observation.state"]
        chunk[:, 1] = 999
        return chunk

    def select_action(self, batch):
        raise AssertionError("offline validation must not use the stateful select_action API")


def test_compute_action_rmse_reduces_action_dimension():
    predicted = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    truth = torch.zeros_like(predicted)
    expected = torch.tensor([(25 / 2) ** 0.5, 0.0]).numpy()
    assert (compute_action_rmse(predicted, truth) == expected).all()


def test_validate_uses_first_action_from_stateless_chunk_api():
    dataset = [
        {"observation.state": torch.tensor([1.0, 2.0]), "action": torch.tensor([[1.0, 2.0], [8.0, 8.0]])},
        {"observation.state": torch.tensor([3.0, 4.0]), "action": torch.tensor([[3.0, 4.0], [8.0, 8.0]])},
        {"observation.state": torch.tensor([5.0, 6.0]), "action": torch.tensor([[5.0, 6.0], [8.0, 8.0]])},
    ]
    cfg = SimpleNamespace(
        batch_size=2,
        num_workers=0,
        policy=SimpleNamespace(use_amp=False),
        action_success_threshold=0.1,
        save_predictions=True,
    )
    policy = _OfflinePolicy()
    result = validate_on_dataset(dataset, policy, cfg, torch.device("cpu"))
    assert policy.chunk_calls == 2
    assert result["aggregated"]["total_samples"] == 3
    assert result["aggregated"]["overall_success_rate"] == 100
    assert result["predictions"]["predicted_actions"].shape == (3, 2)
