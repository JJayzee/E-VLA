# Event Adapter architecture

`SmolVLAPolicy` combines SmolVLA with an event encoder and cross-modal fusion.
Inputs are image, events, language instruction, and robot state; the output is an
action chunk.

## Model

SmolVLM2's SigLIP encoder processes the image. `event_repr.py` converts raw events
to a three-channel representation, which is encoded by `EventEncoder`.
`CrossModalFusion` injects event features into configured SigLIP layers. Language,
visual tokens, and state form the prefix; noisy actions and time embeddings form
the suffix used by the flow-matching action expert.

Key configuration fields:

- `use_event_encoder`;
- `event_repr`: `accu_count` or `accu_sum`;
- `event_fusion_layers`;
- `event_window_size` or `event_window_duration`;
- `train_event_branch_only` and `train_expert_and_event_branch`.

## Training

`LeRobotEventDataset` aligns videos, event NPZ files, and exposure-timestamp
sidecars. It slices and pads event windows. The policy constructs image/event
representations, normalizes inputs, and computes the flow-matching loss.

The event adapter is initialized from a compatible image-only SmolVLA checkpoint.
Training can freeze the base policy or jointly optimize the event branch and action
expert according to the configuration.

## Inference

Online observations include `observation.timestamps.DAVIS`. The policy buffers raw
events and applies the checkpoint window at the exposure end time. Dataset samples
omit exposure timestamps and provide complete event windows.

`reset()` clears the action queue and event buffer and must be called at every
episode boundary.
