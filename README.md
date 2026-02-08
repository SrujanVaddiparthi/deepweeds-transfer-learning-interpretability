# Weed Species Classification (Transfer Learning & Interpretability)

This project reproduces and extends the **DeepWeeds** weed-classification benchmark using a modern PyTorch workflow, with a focus not just on accuracy, but on **model interpretability and biological relevance**.

The motivation behind this work is simple:  
if machine-learning models are ever to guide real-world actions (e.g., robotic weed removal or UAV-based monitoring), we must understand *why* they make certain predictions and *what visual structures they rely on*.

---

## Project Context

- **Course:** Applications of Machine Learning for Remote Sensing  
- **Institution:** Rochester Institute of Technology  
- **Focus:** Transfer learning, robustness, and explainable computer vision

This project builds on the **DeepWeeds** dataset (RGB imagery collected in natural rangeland environments) and treats it as a *proximal remote sensing* problem, where models must operate under cluttered backgrounds and real-world variability.

---

## Problem Statement

While prior work shows that CNNs can classify weed species with high accuracy, two key questions remain:

1. Are these results stable beyond a single random train/test split?
2. Do models attend to **biologically meaningful plant structures**, or are predictions driven by spurious background cues?

This project addresses both.

---

## Approach

### 1. Reproducible Classification Pipeline
- Parsed COCO-style Weed-AI annotations into a structured PyTorch dataset
- Implemented stratified train/validation/test splits
- Trained an ImageNet-pretrained **ResNet-50** using targeted augmentation and early stopping

### 2. Robustness Evaluation
- Trained across multiple randomized splits and seeds
- Verified that test accuracy (~93–94%) was **stable**, not a single-run artifact

### 3. Error Analysis
- Analyzed confusion matrices and per-class metrics
- Identified intuitive failure modes driven by:
  - background clutter
  - visually similar weed species
  - partial occlusions

### 4. Interpretability with Grad-CAM
- Applied Grad-CAM to the final convolutional layers
- Visualized spatial attention to assess whether the model focuses on:
  - leaf clusters
  - branching patterns
  - crown-adjacent regions  
  rather than irrelevant background textures

---

## Key Results

- Achieved **~94 percent test accuracy** using transfer learning
- Demonstrated **performance stability** across multiple random splits
- Showed that Grad-CAM activations align with biologically meaningful plant structures
- Highlighted limitations of RGB-only classification for precise crown localization

---

## Limitations & Future Directions

This project is intentionally scoped as a **classification and interpretability study**.

Identified extensions include:
- Crown-center localization using weak supervision
- Lightweight segmentation for robotic targeting
- Fusion with UAV imagery or multi-spectral data
- Integration into field-deployable robotic or drone systems

The broader goal is to bridge **perception and action** — enabling interpretable ML models that can guide real-world decisions in agriculture and environmental monitoring.

---

## Repository Structure

