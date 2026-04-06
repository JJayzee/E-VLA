# E-VLA: Event-Augmented Vision-Language-Action Model for Dark and Blurred Scenes

This is the official repository for **E-VLA**, a framework that integrates event-driven perception into Vision-Language-Action models to enhance robotic manipulation robustness under adverse sensing conditions.

---

## 🎥 Video Demo


https://github.com/user-attachments/assets/937e8a7d-80f2-473d-9c40-4e3810e36c87


> **Highlights:** E-VLA maintains high success rates in extreme low-light (20 lux) and severe motion blur scenarios where conventional frame-based VLA models typically fail.

---

## 🛠️ Code and Dataset

The paper is currently **under review**. The codebase and the real-world RGB-Event-Action dataset will be made available in this repository as the project progresses. 

---

## 🚀 Key Contributions

* **Robust Perception:** Leverages high dynamic range and high temporal resultion events to preserve perception consistency during degradations like low light and motion blur.
* **Efficient Integration:** Proposes lightweight, pretrained-compatible event integration strategies for stable deployment.
* **Dataset:** Collects a synchronized multi-modal manipulation dataset under diverse light conditions using the DAVIS346 event camera and LeRobot SO100 manipulator.
* **Experimental Evidence:** Demonstrates significant performance gains (above 25% in low-light and blurred scenes, 80% in extremely low light) across various tasks.

---

## 🤝 Acknowledgements

Thanks to the following amazing works from the open-source community:

* [**SmolVLA**](https://huggingface.co/blog/smolvla): For the efficient VLA base model and pre-training insights.
* [**LeRobot**](https://github.com/huggingface/lerobot): For the LeRobot framework and SO10x kits that facilitated our data collection and robot deployment.
