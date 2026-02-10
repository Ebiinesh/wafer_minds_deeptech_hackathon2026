# Wafer Minds - Edge-AI Wafer Defect Classification (Carinthia Dataset)

## Project Overview

This repository presents an end-to-end Edge-AI pipeline for semiconductor wafer defect classification, developed to enable automated, low-latency inspection in smart manufacturing environments. Semiconductor fabrication processes generate a large number of high-resolution inspection images at multiple stages, and manual or centralized inspection approaches often struggle to scale due to latency, bandwidth, and infrastructure constraints. The objective of this project is to demonstrate how deep learning can be applied to classify wafer defects accurately while remaining suitable for future deployment on edge devices.

The scope of this work aligns with Phase-1 of the hackathon and focuses on dataset engineering, defect reclassification, model training using transfer learning, quantitative evaluation, and preparation of an edge-compatible model. All experimentation and development were performed in a cloud-based Google Colab environment, and a recorded Colab simulation video demonstrating the full workflow is attached for clarity and transparency.

**Project Presentation (PDF)**  
[Wafer Minds_PS01.pdf](./Wafer_Minds_PS01.pdf)

---

## 📂 Dataset Description and Reclassification

This project exclusively uses the **Carinthia SEM wafer defect dataset**, a publicly available dataset containing scanning electron microscope (SEM) images of real semiconductor defects. The dataset captures fine-grained structural and surface-level defect patterns commonly observed in wafer fabrication, making it highly relevant for industrial inspection and yield analysis research.

The original dataset is publicly available at:  
**https://zenodo.org/records/10715190**

Rather than directly using the raw labels, the dataset was **reclassified and reorganized into eight application-relevant defect categories** to better reflect real fab inspection scenarios and to satisfy hackathon requirements. The final classes used in this work are:

- Clean  
- Scratch  
- Crack  
- Particle  
- Line_Edge_Roughness  
- Pattern_Misalignment  
- Surface_Nonuniformity  
- Other  

Rare or under-represented defect types were consolidated into the **Other** category, while highly frequent pattern-related defects were preserved as separate classes. This restructuring ensures meaningful classification while maintaining realistic industrial data characteristics.

The dataset was split into **Train and Test partitions using an 80:20 ratio**, preserving class distribution as much as possible.

### Training Set Distribution (3,849 images total)

- Clean: 180  
- Crack: 181  
- Line_Edge_Roughness: 801  
- Pattern_Misalignment: 801  
- Surface_Nonuniformity: 801  
- Particle: 231  
- Scratch: 44  
- Other: 810  

### Test Set Distribution (962 images total)

- Clean: 40  
- Crack: 46  
- Line_Edge_Roughness: 201  
- Pattern_Misalignment: 201  
- Surface_Nonuniformity: 201  
- Particle: 58  
- Scratch: 11  
- Other: 204  

These statistics reflect a realistic semiconductor inspection dataset, where certain defect types are naturally rare while others appear more frequently due to process variations.

---

## 🧠 Model Development

Model development was carried out using **PyTorch**, adopting a transfer learning strategy to balance classification accuracy with computational efficiency. A lightweight convolutional backbone (EfficientNet-B0) was selected due to its strong performance-to-parameter ratio, making it suitable for future edge deployment.

All SEM images were converted to grayscale and resized to **224 × 224** resolution. To maintain compatibility with pretrained CNN architectures, the single channel images were replicated across three channels. Training was performed in two stages: an initial feature extraction phase with frozen backbone layers, followed by fine-tuning to improve adaptation to the semiconductor defect domain. Cross-entropy loss with class weighting and the Adam optimizer were used to ensure stable convergence.

---

## 📊 Quantitative Results and Analysis

The trained model was evaluated on a held-out test set of 962 images across eight defect classes. The overall performance demonstrates strong generalization despite dataset imbalance.

**Overall Performance Metrics:**
- **Accuracy:** **94.3%**
- **Macro Average F1-score:** 0.910
- **Weighted Average Precision:** 0.9464
- **Weighted Average Recall:** 0.9428
- **Weighted Average F1-score:** 0.9437

Class-wise results indicate that structurally dominant defects such as **Line_Edge_Roughness**, **Pattern_Misalignment**, and **Surface_Nonuniformity** achieved F1-scores above 0.94, showing consistent discrimination capability. The **Particle** class also exhibited strong performance with an F1-score of 0.942. As expected in industrial datasets, the **Scratch** class, being rare, showed lower precision but perfect recall, indicating that the model successfully avoided missing critical defects.

A full classification report and confusion matrix are included in this repository to provide detailed insight into model behavior across classes.

---

## 🛠 Tools, Frameworks, and Environment

The complete development and evaluation pipeline was built using the following tools and frameworks:

- **Development Environment:** Google Colab (GPU-accelerated)
- **Programming Language:** Python 3.x
- **Deep Learning Framework:** PyTorch
- **Data Processing:** Pandas, NumPy
- **Image Processing:** OpenCV / PIL
- **Evaluation Metrics:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Model Interoperability:** ONNX (for edge compatibility)

The trained model was exported in **ONNX format**, ensuring readiness for integration with the **NXP eIQ toolkit** and future deployment on **i.MX RT series microcontrollers**.

---

## Code Reference (Google Colab)

All dataset processing, training, evaluation, and model export code is provided in the following Google Colab notebook:

🔗 **Colab Notebook**  
https://colab.research.google.com/drive/1E6kSrdddTTrwZJwVnAr3F6gfQ6EZjZiP?usp=sharing

This notebook serves as the primary reference for the complete implementation.

---

## 🎥 Google Colab Simulation Video

A complete **Google Colab simulation video** demonstrating the full workflow — dataset preparation, reclassification, training, evaluation, and result generation — is available at:

🎥 **Simulation Video**  
https://drive.google.com/file/d/1m2CR6tNh1pl29nN5pMSDvudTbNMbTQoC/view?usp=sharing

This video provides visual validation of the experiments and enhances transparency for reviewers and collaborators.

---

## Exported ONNX Model

The trained model has been exported to **ONNX** format for cross-framework compatibility, inference optimization, and deployment in various runtimes (e.g., ONNX Runtime, TensorRT, etc.).

🔗 **ONNX Model File**  
[https://drive.google.com/file/d/1UZwahQyMnELbh9_TrhNOE6htftZ1S9Ec/view?usp=sharing](https://drive.google.com/drive/folders/1_yRfhZ8glCu1s9-ywj0VjqI1mwEMGA7Q?usp=sharing)

*(File name appears to be `wafer_defect_model.onnx`)*

---

## Project Google Drive Folder**  
[Open Project Folder](https://drive.google.com/drive/folders/1Yoripvb_VOPmAt01qyUrkY0ZKxICiz_s?usp=sharing)

---

This completes the set of resources:  
- Full reproducible code in Colab  
- Step-by-step walkthrough video  
- Portable ONNX model export ready for deployment

*Submitted for DeepTech Hackathon 2026.*

---

