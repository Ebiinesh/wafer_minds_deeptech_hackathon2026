# Edge-AI Wafer Defect Classification (Carinthia Dataset)

## 📌 Project Overview

This repository contains the implementation and artifacts for an Edge‑AI
based wafer defect classification system. The goal of this project is to
analyze high‑resolution inspection images from semiconductor wafer
fabrication and automatically classify them into multiple defect
categories. Traditional defect analysis methods often rely on manual
review or centralized cloud processing, which introduce latency and
infrastructure overhead. To address this, our solution trains a deep
learning model capable of identifying and categorizing wafer defects
efficiently, with portability toward edge deployment using embedded AI
frameworks such as NXP eIQ.

## 📂 Dataset Description

For data, we exclusively used the **Carinthia SEM defect dataset**, a
publicly available dataset of scanning electron microscope images that
depicts various real semiconductor defects. This dataset captures a rich
variety of defect patterns at the micro scale, making it well‑suited for
training and evaluating deep learning models for inspection
applications. The Carinthia dataset images were carefully relabeled into
eight defect classes relevant to fab inspection workflows: **Clean,
Scratch, Crack, Particle, Line Edge Roughness, Pattern Misalignment,
Surface Nonuniformity, and Other**.

The dataset was organized into **Train, Validation, and Test** splits,
ensuring that the model is trained and evaluated in a reproducible and
statistically sound manner. The original dataset can be accessed here:\
https://zenodo.org/records/10715190

## 🧠 Model & Evaluation

In the model development phase, we employed transfer learning with a
lightweight convolutional backbone to achieve strong classification
performance while maintaining model efficiency, a key requirement for
future edge deployment. The model is trained on grayscale images resized
to a standardized input resolution, and class weighting along with data
augmentation techniques were used to improve generalization. After
training, the model was evaluated on a held‑out test set, and metrics
such as accuracy, precision, recall, and confusion matrix were generated
to quantify performance across the eight defect classes. These
evaluation artifacts are included in the repository to support clear
understanding of model behavior across categories.

## 🎥 Google Colab Demonstration

To support reproducibility and interactive exploration, we have attached
a **Google Colab simulation video** that demonstrates the entire
workflow --- from dataset preparation to training, evaluation, and model
export. This video walkthrough provides visual confirmation of the
experiments and makes it easy for reviewers and teammates to follow the
implementation steps in a cloud environment without requiring local
setup.

## 🚀 Edge Readiness

The trained model is exported in **ONNX format**, enabling compatibility
with the NXP eIQ toolkit and future deployment on i.MX RT series
microcontrollers. This positions the solution for real‑time, on‑device
wafer defect classification in smart manufacturing environments.
