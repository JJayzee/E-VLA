# E-VLA: Event-Augmented Vision-Language-Action Model for Dark and Blurred Scenes

This is the official repository for **E-VLA**, a framework that introduces high-dynamic-range and high-temporal-resolutionevent event signals into VLA models, enabling robots to see and act in dark and blurred scenarios for robust manipulation.

---

## 🎥 Video Demo


https://github.com/user-attachments/assets/937e8a7d-80f2-473d-9c40-4e3810e36c87


> **Highlights:** E-VLA maintains high success rates under extremely low-light (3/255 average grayscale or 4lux) and severe motion-blurred conditions where conventional frame-based VLA models typically fail.

---

## 🛠️ Code and Dataset

The codebase and the real-world RGB-Event-Action dataset will be made available in this repository as the project progresses. 

---

## 🚀 Key Contributions

* **Robust Perception:** Leverages high dynamic range and high temporal resultion events to preserve perception consistency during degradations like low light and motion blur.
* **Efficient Integration:** Proposes lightweight, pretrained-compatible event integration strategies for stable deployment.
* **Dataset:** Collects a synchronized multi-modal manipulation dataset under diverse light conditions using the DAVIS346 event camera and LeRobot SO100 manipulator.
* **Experimental Evidence:** Demonstrates significant performance gains (~25% in low-light and blurred scenes, 90% in extremely low-light scenes and 36% in OOD settings).

---

## 🤝 Acknowledgements

Thanks to the following amazing works from the open-source community:

* [**SmolVLA**](https://huggingface.co/blog/smolvla): For the efficient VLA base model and pre-training insights.
* [**LeRobot**](https://github.com/huggingface/lerobot): For the LeRobot framework and SO10x kits that facilitated our data collection and robot deployment.
