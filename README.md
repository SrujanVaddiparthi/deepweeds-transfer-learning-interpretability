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

notebooks/
  EDA.ipynb
  training_and_evaluation.ipynb
data/
  (not included)
environment.yml
EDA.ipynb - jupyter notebook used to develop the project
report/
  Fine-Tuning CNNs for Weed Species Classification.pdf
README.md


---

## Setup

``bash
conda env create -f env.yml
conda activate weed-classification
jupyter notebook


---


## References

1. Chattopadhyay, A. (2018). Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep 
Convolutional Networks. 2018 IEEE Winter Conference on Applications of Computer Vision 
(WACV). https://doi.org/10.1109/WACV.2018.00097 

2. He, K. (2016). Deep Residual Learning for Image Recognition. arXiv. 
https://doi.org/10.48550/arXiv.1512.03385 

3. Kirillov, A. (2023). Segment Anything. https://arxiv.org/abs/2304.02643 

4. Kundu, N. (2021). IoT and Interpretable Machine Learning Based Framework for Disease Prediction in 
Pearl Millet. National Library of Medicine. 

5. Muhammad, M. (2021). Eigen-CAM: Visual Explanations for Deep Convolutional Neural Networks. 
10.1007/s42979-021-00449-3 

6. Olsen, S. R. (2019). DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning. Scientific 
Report. https://doi.org/10.1038/s41598-018-38343-3 

7. Precision Weed Control Group and Sydney Informatics Hub, the University of Sydney. (n.d.). Weed-AI: A 
repository of Weed Images in Crops. Weed-AI: A repository of Weed Images in Crops. 
https://weed-ai.sydney.edu.au/ 

8. PyTorch. (n.d.). Resnet ImageNet normalization. 
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvi
sion.models.ResNet50_Weights 

9. Selvaraju, R. R. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based 
Localization. IEEE Xplore. 10.1109/ICCV.2017.74 

10. Singhi, S. (2024). Strengthening Interpretability: An Investigative Study of Integrated Gradient Methods. 
https://arxiv.org/html/2409.09043v1 

11. Zhang, J. (2020). Segmenting Purple Rapeseed Leaves in the Field from UAV RGB Imagery Using Deep 
Learning as an Auxiliary Means for Nitrogen Stress Detection. ResearchGate. 
10.3390/rs12091403 






