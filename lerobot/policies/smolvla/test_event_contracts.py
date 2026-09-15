from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.cameras.davis_utils import apply_white_balance
from lerobot.datasets.lerobot_event_dataset import EVENT_DTYPE, _canonicalize_stored_events
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.event_repr import _events_to_image, _events_to_image_by_accu_and_bayer
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


@pytest.mark.parametrize("representation", ["accu_count", "accu_sum"])
def test_event_representations_shape_dtype_and_range(representation):
    events = torch.tensor([[10, 20, 1, 100], [10, 20, 0, 101], [11, 21, 1, 102]])
    image = _events_to_image(events, representation)
    assert image.shape == (3, 260, 346)
    assert image.dtype == torch.float32
    assert torch.isfinite(image).all()
    assert 0 <= image.min() <= image.max() <= 1


def test_empty_event_representations_have_expected_baseline():
    events = torch.empty((0, 4), dtype=torch.int32)
    assert torch.count_nonzero(_events_to_image(events, "accu_count")) == 0
    sum_image = _events_to_image(events, "accu_sum")
    assert torch.allclose(sum_image, torch.full_like(sum_image, 127 / 255))


@pytest.mark.parametrize("events", [torch.tensor([[-1, 0, 1, 0]]), torch.tensor([[346, 0, 1, 0]]), torch.tensor([[0, 260, 1, 0]])])
def test_event_representation_rejects_out_of_bounds_coordinates(events):
    with pytest.raises(ValueError, match="Event coordinates"):
        _events_to_image(events, "accu_count")


def test_event_representation_rejects_bad_shape_and_mode():
    with pytest.raises(ValueError, match="shape"):
        _events_to_image_by_accu_and_bayer(torch.zeros((2, 3)), "count")
    with pytest.raises(ValueError, match="Unknown accumulation mode"):
        _events_to_image_by_accu_and_bayer(torch.zeros((2, 4)), "bad")
    with pytest.raises(ValueError, match="Unknown event representation"):
        _events_to_image(torch.zeros((2, 4)), "bad")


def test_dense_and_structured_events_canonicalize_without_mutating_input():
    dense = torch.tensor([[2, 3, 1, 123]], dtype=torch.int64).numpy()
    original = dense.copy()
    canonical = _canonicalize_stored_events(dense)
    assert canonical.dtype == EVENT_DTYPE
    assert (canonical["x"], canonical["y"], canonical["polarity"], canonical["timestamp"]) == (2, 3, 1, 123)
    assert (dense == original).all()

    structured = canonical.copy()
    converted = _canonicalize_stored_events(structured)
    assert converted.dtype == EVENT_DTYPE
    assert converted is not structured


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"event_window_size": 1, "event_window_duration": 1}, "mutually exclusive"),
        ({"event_window_size": 0}, "positive integer"),
        ({"event_window_duration": 0}, "positive duration"),
    ],
)
def test_event_window_config_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SmolVLAConfig(**kwargs)


def _lightweight_policy(event_window_size=2, event_window_duration=None):
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    torch.nn.Module.__init__(policy)
    image_key = "observation.images.DAVIS"
    event_key = "observation.events.DAVIS"
    policy.config = SimpleNamespace(
        device="cpu",
        image_features={image_key: SimpleNamespace(shape=(3, 260, 346))},
        event_features={event_key: SimpleNamespace(shape=(None, 4))},
        event_window_size=event_window_size,
        event_window_duration=event_window_duration,
        n_action_steps=2,
        adapt_to_pi_aloha=False,
    )
    policy.normalize_inputs = lambda batch: batch
    policy._event_buffer = torch.empty((0, 4), dtype=torch.int32)
    policy.reset()
    return policy


@pytest.mark.parametrize(
    "image,expected_shape",
    [
        (np.zeros((260, 346, 3), dtype=np.uint8), (1, 3, 260, 346)),
        (torch.zeros((3, 260, 346)), (1, 3, 260, 346)),
        (torch.zeros((2, 3, 260, 346)), (2, 3, 260, 346)),
        (torch.zeros((2, 4, 3, 260, 346)), (2, 4, 3, 260, 346)),
    ],
)
def test_canonicalize_image_layouts_and_auto_batch(image, expected_shape):
    policy = _lightweight_policy()
    source = {
        "observation.images.DAVIS": image,
        "observation.state": np.zeros(6, dtype=np.float32),
        "observation.events.DAVIS": np.zeros((3, 4), dtype=np.int64),
    }
    converted = policy._canonicalize_inference_batch(source)
    assert converted["observation.images.DAVIS"].shape == expected_shape
    assert converted["observation.images.DAVIS"].dtype == torch.float32
    assert converted["observation.state"].shape == (1, 6)
    assert converted["observation.events.DAVIS"].shape == (3, 4)
    assert converted["observation.events.DAVIS"].dtype == torch.int32
    assert source["observation.images.DAVIS"] is image


def test_online_davis_image_uses_offline_lamp_white_balance_contract():
    policy = _lightweight_policy()
    image = np.empty((260, 346, 3), dtype=np.uint8)
    image[...] = [32, 64, 96]
    original = image.copy()
    source = {
        "observation.images.DAVIS": image,
        "observation.state": np.zeros(6, dtype=np.float32),
        "observation.events.DAVIS": np.empty((0, 4), dtype=np.int32),
    }

    converted = policy._canonicalize_inference_batch(source)
    expected = apply_white_balance(
        image, np.array([1.4066648, 0.6607546, 1.2891957])
    )
    expected = torch.from_numpy(expected).permute(2, 0, 1).unsqueeze(0).float() / 255

    assert torch.equal(converted["observation.images.DAVIS"], expected)
    assert np.array_equal(image, original)


def test_online_count_window_deduplicates_and_reset_clears_state():
    policy = _lightweight_policy(event_window_size=2)
    captured = []

    def fake_action_chunk(self, batch, noise=None):
        captured.append(batch["observation.events.DAVIS"].clone())
        return torch.zeros((1, 2, 6))

    policy._get_action_chunk = MethodType(fake_action_chunk, policy)
    base = {
        "observation.images.DAVIS": np.zeros((260, 346, 3), dtype=np.uint8),
        "observation.state": np.zeros(6, dtype=np.float32),
    }
    first = {**base, "observation.events.DAVIS": np.array([[0, 0, 1, 1], [1, 1, 1, 2], [2, 2, 0, 3]]), "observation.timestamps.DAVIS": [0, 3]}
    original_events = first["observation.events.DAVIS"].copy()
    policy.select_action(first)
    assert captured[0][0, :, 3].tolist() == [2, 3]
    assert np.array_equal(first["observation.events.DAVIS"], original_events)

    second = {**base, "observation.events.DAVIS": np.array([[1, 1, 1, 2], [2, 2, 0, 3], [3, 3, 1, 4], [4, 4, 1, 5]]), "observation.timestamps.DAVIS": [3, 5]}
    policy.select_action(second)
    assert policy._event_buffer[:, 3].tolist() == [4, 5]
    policy.reset()
    assert policy._event_buffer.shape == (0, 4)
    assert len(policy._queues["action"]) == 0


def test_predict_action_chunk_rejects_online_timestamps():
    policy = _lightweight_policy()
    batch = {
        "observation.images.DAVIS": np.zeros((260, 346, 3), dtype=np.uint8),
        "observation.state": np.zeros(6, dtype=np.float32),
        "observation.events.DAVIS": np.empty((0, 4), dtype=np.int32),
        "observation.timestamps.DAVIS": [0, 1],
    }
    with pytest.raises(ValueError, match="already-windowed"):
        policy.predict_action_chunk(batch)
