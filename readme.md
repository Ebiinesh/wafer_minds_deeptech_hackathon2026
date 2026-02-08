# DEFECTIQ --- Edge-AI Wafer Defect Classification (Carinthia Dataset)

## 📌 Project Overview

This repository presents an end-to-end Edge‑AI pipeline for
semiconductor wafer defect classification, developed for automated,
low‑latency inspection in smart manufacturing environments. The
objective of this work is to replace slow, centralized, and manual
inspection workflows with a data‑driven, on‑device artificial
intelligence solution capable of identifying and categorizing wafer
defects directly at the point of capture. By leveraging lightweight deep
learning models and edge-compatible formats, the project demonstrates
how modern AI techniques can be integrated into semiconductor
fabrication quality control while remaining compatible with embedded
deployment platforms such as NXP eIQ and i.MX RT devices.

The implementation covers the full workflow required in Phase‑1 of the
hackathon: dataset curation and restructuring, model training using
transfer learning, rigorous quantitative evaluation, and preparation of
an edge‑ready ONNX model. All experiments were carried out in a
cloud-based Google Colab environment, and a recorded Colab simulation
video demonstrating the complete pipeline is attached in this repository
for transparency and reproducibility.

------------------------------------------------------------------------

## 📂 Dataset Description and Reorganization

For this work, we exclusively used the **Carinthia SEM wafer defect
dataset**, a publicly available collection of high‑resolution scanning
electron microscope (SEM) images that capture realistic defect patterns
observed in semiconductor manufacturing. The dataset contains diverse
visual characteristics such as micro‑scratches, surface texture
variations, particulate contamination, and structural irregularities,
making it highly relevant for industrial inspection research. The
original dataset is available at:\
https://zenodo.org/records/10715190

Rather than using the raw labels as provided, we **restructured and
reclassified the dataset into eight application‑relevant categories**
aligned with real fab inspection terminology and hackathon requirements.
The final classes were: **Clean, Scratch, Crack, Particle,
Line_Edge_Roughness, Pattern_Misalignment, Surface_Nonuniformity, and
Other.** This mapping involved analyzing the original defect
distributions, merging under‑represented categories into "Other," and
redistributing highly frequent patterns into multiple semantically
meaningful defect types.

The reorganized dataset was split into **Train and Test partitions in an
80:20 ratio**, preserving class distribution as much as possible. The
final dataset statistics were:

**Training Set (3,849 images total):**\
- Clean: 180\
- Crack: 181\
- Line_Edge_Roughness: 801\
- Pattern_Misalignment: 801\
- Surface_Nonuniformity: 801\
- Particle: 231\
- Scratch: 44\
- Other: 810

**Test Set (962 images total):**\
- Clean: 40\
- Crack: 46\
- Line_Edge_Roughness: 201\
- Pattern_Misalignment: 201\
- Surface_Nonuniformity: 201\
- Particle: 58\
- Scratch: 11\
- Other: 204

These numbers demonstrate a realistic, slightly imbalanced industrial
dataset, where certain defect types (e.g., scratches) are naturally rare
compared to more frequent structural variations such as line roughness
or surface nonuniformity.

------------------------------------------------------------------------

## 🧠 Model Development and Training

Model development was performed using **PyTorch**, selecting a
lightweight yet high‑performing convolutional backbone suitable for edge
deployment. We adopted a **transfer learning strategy with
EfficientNet‑B0**, modifying the final classification layer to output
eight defect classes. This approach allowed us to leverage pretrained
visual representations while adapting the model to the specific
characteristics of SEM wafer images.

All images were converted to grayscale and resized to **224 × 224**,
then replicated across three channels to maintain compatibility with
standard CNN architectures. Training was conducted in two stages: (1) a
feature‑extraction phase with the backbone frozen, followed by (2)
fine‑tuning of deeper layers to improve domain adaptation. We used
**Cross‑Entropy Loss** with class weighting to mitigate dataset
imbalance and the **Adam optimizer** with a learning rate schedule for
stable convergence.

Training and evaluation were conducted entirely in **Google Colab**,
utilizing GPU acceleration to reduce runtime. The entire process---from
dataset loading to model export---has been recorded in an attached
**Google Colab simulation video**, enabling reviewers to visually trace
every step of the pipeline.

------------------------------------------------------------------------

## 📊 Quantitative Results

On the held‑out test set of 962 images, the model achieved strong
classification performance across all eight classes. The key results
were:

-   **Overall Accuracy:** **94.3%**\
-   **Macro Average F1‑score:** 0.910\
-   **Weighted Average Precision:** 0.9464\
-   **Weighted Average Recall:** 0.9428\
-   **Weighted Average F1‑score:** 0.9437

Class‑wise performance showed that common defect categories such as
**Line_Edge_Roughness, Pattern_Misalignment, and Surface_Nonuniformity**
achieved F1‑scores above 0.94, indicating reliable discrimination. The
**Particle** class also performed strongly with an F1‑score of 0.942. As
expected in real-world datasets, the **Scratch** class---being
rare---exhibited lower precision (0.579) but perfect recall (1.000),
suggesting the model was conservative in detecting this defect type
rather than missing instances.

A full confusion matrix and classification report are included in the
repository, providing transparency into inter-class misclassifications
and model behavior.

------------------------------------------------------------------------

## 🛠 Tools, Frameworks, and Deployment Readiness

The technical stack used in this project was designed with
reproducibility, scalability, and edge compatibility in mind:

**Development Environment:**\
- **Google Colab** (GPU-enabled cloud environment for training and
experimentation)

**Programming Language:**\
- **Python 3.10+**

**Deep Learning Framework:**\
- **PyTorch** for model training, evaluation, and experimentation

**Data Processing Libraries:**\
- **Pandas** for CSV parsing and dataset organization\
- **NumPy** for numerical operations\
- **OpenCV / PIL** for image preprocessing and resizing\
- **Scikit-learn** for metric computation and classification reports

**Model & Edge Format:**\
- The trained model was exported to **ONNX format**, ensuring
compatibility with the **NXP eIQ Toolkit** and future deployment on
**i.MX RT series microcontrollers**. This step positions the solution
for real-time, on-device inference in smart semiconductor fabs.

**Visualization & Reporting:**\
- **Matplotlib and Seaborn** were used to generate confusion matrices
and performance visualizations.

By integrating these tools, the project establishes a complete pipeline
from dataset engineering to edge-ready AI deployment, demonstrating both
technical rigor and practical applicability for industrial semiconductor
inspection.

------------------------------------------------------------------------

## 🎥 Colab Demonstration

A **Google Colab simulation video** is attached to this repository,
illustrating the complete workflow: dataset loading, preprocessing,
train-test split, model training, evaluation, and ONNX export. This
serves as both documentation and validation of the experimental process.
