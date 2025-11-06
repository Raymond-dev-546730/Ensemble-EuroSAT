# Ensemble Learning for Land Cover Classification on EuroSAT

This repository contains training scripts and experimental logs from:

> **"Ensemble Learning for Land Cover Classification: Advancing State-of-the-Art Performance on EuroSAT"**  
> *Accepted to International Joint Conference on Neural Networks (IJCNN) 2025*  
> Raymond Lee & Ethan Knapp

## Overview

We present an ensemble learning approach for satellite image classification that achieves **99.87±0.02% accuracy** on the EuroSAT benchmark dataset, a **0.46±0.02% improvement** over previous state-of-the-art models. Our method combines three convolutional neural networks (ResNet50, RegNet-Y-8GF, and EfficientNetV2-S) with a final ensemble network for land cover classification across 10 categories.

## Repository Contents

~~~
Scripts/
├── EuroSAT_Pre-processing.py           # Data preprocessing pipeline
├── ResNet50_EuroSAT.py                 # ResNet50 base model training
├── RegNet-Y-8GF_EuroSAT.py             # RegNet-Y-8GF base model training
├── EfficientNetV2-S_EuroSAT.py         # EfficientNetV2-S base model training
└── Ensemble_Model_EuroSAT.py           # Ensemble model training

Terminal_Logs/
├── ResNet50 Terminal Logs.txt          # ResNet50 training outputs
├── RegNet-Y-8GF Terminal Logs.txt      # RegNet-Y-8GF training outputs
├── EfficientNetV2-S Terminal Logs.txt  # EfficientNetV2-S training outputs
└── Ensemble Model Terminal Logs.txt    # Ensemble model training outputs
~~~

## Key Results

- **Accuracy:** 99.87±0.02%
- **F1-Score:** 99.86±0.02%
- **Precision:** 99.86±0.02%
- **Recall:** 99.87±0.02%
- **ROC AUC:** 99.99±0.00%

As of May 2025, our ensemble model surpasses all previous benchmarks on the EuroSAT dataset, including DeepEnsembling (99.41%), IMP+MTP (99.24%), and µ2Net+ (99.22%).

## Dataset

The EuroSAT dataset is available at: https://www.kaggle.com/datasets/apollo2506/eurosat-dataset 

## Citation

If you use this code or reference our work, please cite:
```bibtex
@inproceedings{lee2025ensemble,
  title={Ensemble Learning for Land Cover Classification: Advancing State-of-the-Art Performance on EuroSAT},
  author={Lee, Raymond and Knapp, Ethan},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  year={2025}
}
```
