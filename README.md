# E-VLA: Event-Augmented Vision-Language-Action Model for Dark and Blurred Scenes

**Accepted to ECCV 2026.** [Paper](https://arxiv.org/abs/2604.04834)

E-VLA augments vision-language-action models with event-camera signals to support robotic manipulation in dark and motion-blurred scenes. It uses synchronized RGB, events, and actions collected with a DAVIS346 camera and an SO-100 manipulator.

## Video demo

https://github.com/user-attachments/assets/937e8a7d-80f2-473d-9c40-4e3810e36c87

> E-VLA was evaluated under extremely low light and severe motion blur, where frame-only perception can become unreliable.

## Code and dataset

The implementation is organized into two branches:

| Branch | Contents | Guide |
| --- | --- | --- |
| [`event_adapter`](https://github.com/JJayzee/E-VLA/tree/event_adapter) | Synchronized DAVIS capture, event-aware datasets, event-adapter training, and deployment | [Event-adapter guide](https://github.com/JJayzee/E-VLA/blob/event_adapter/docs/event_adapter.md) |
| [`image_only`](https://github.com/JJayzee/E-VLA/tree/image_only) | Image-only SmolVLA baseline, multitask training, and image video profiles | [Image-only guide](https://github.com/JJayzee/E-VLA/blob/image_only/docs/image_only.md) |

The synchronized RGB-event-action dataset is available at [JJayzee/E-VLA-MultiLight](https://huggingface.co/datasets/JJayzee/E-VLA-MultiLight). Dataset layout and feature contracts are documented in each branch's guide.

## Key contributions

- **Robust perception:** Event streams retain motion and structural cues when RGB frames are degraded by low light or blur.
- **Efficient integration:** Event representations and lightweight fusion strategies extend a pretrained SmolVLA policy.
- **Synchronized data:** A DAVIS346 camera and SO-100 manipulator provide RGB, event, timestamp, action, and task data across illumination conditions.
- **Experimental evidence:** The [paper](https://arxiv.org/abs/2604.04834) reports manipulation results for image-only and event-augmented policies under adverse visual conditions.

## Citation

```bibtex
@inproceedings{zhai2026evla,
  author = {Zhai, Jiajun and Shi, Hao and Guo, Shangwei and Yang, Kailun and Wang, Kaiwei},
  title = {E-VLA: Event-Augmented Vision-Language-Action Model for Dark and Blurred Scenes},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2026}
}
```

## Acknowledgements

E-VLA builds on [SmolVLA](https://huggingface.co/blog/smolvla) and [LeRobot](https://github.com/huggingface/lerobot). We thank their maintainers for the base model, robotics framework, and SO-10x ecosystem.
