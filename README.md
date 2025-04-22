# Edge-Boosted Graph Learning

[**Edge-Boosted Graph Learning for Functional Brain Connectivity Analysis**](https://arxiv.org/abs/2504.14796)  
📄 Published in IEEE ISBI 2025  
📚 Paper: *"EDGE-BOOSTED GRAPH LEARNING FOR FUNCTIONAL BRAIN CONNECTIVITY ANALYSIS"*

## Overview

This repository contains the code implementation for the IEEE ISBI 2025 paper that proposes a novel Graph Neural Network (GNN) framework incorporating **edge-to-edge (eFC)** and **node-to-node** relationships to better analyze functional brain connectivity. Unlike traditional GNNs that focus solely on node attributes, this method co-embeds edge functional connectivity to capture dynamic brain interactions more effectively.

## Key Features

- ✅ Edge Time Series (eTS) calculation to extract co-fluctuation patterns between brain regions  
- ✅ Construction of Edge Functional Connectivity (eFC) matrices  
- ✅ Co-embedding of node and edge attributes using a unified GNN architecture  
- ✅ Performance benchmarking on **ADNI** and **PPMI** fMRI datasets  
- ✅ Significant improvement over CNN, GCN, CRGNN, and MGNN baselines  

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/EdgeBoostedGNN.git
cd EdgeBoostedGNN
pip install -r requirements.txt
```

> ⚠️ Note: Use Python ≥ 3.8 and ensure PyTorch with CUDA is properly configured for GPU training.

## Requirements

All dependencies are listed in `conda_requirements.txt`.  
You can install them using:

```bash
pip install -r requirements.txt
```

## Running the Model

Navigate to the `scripts/` folder and launch training:

```bash
python utill_gpu.py
```

## Datasets

We evaluate on the following datasets:

- [ADNI Dataset](https://adni.loni.usc.edu/)
- [PPMI Dataset](https://www.ppmi-info.org/)

All fMRI scans were preprocessed using `fMRIPrep` and converted into node and edge attribute matrices.

## Model Architecture

- Dual-layer GCN: node aggregation and edge aggregation  
- Co-embedding layer fuses node and edge features  
- Dropout, batch norm, and ReLU activations included

## Performance Summary

| Dataset | Accuracy | Precision | F1 Score |
|---------|----------|-----------|----------|
| ADNI    | **0.8000** ± 0.0876 | 0.8437 ± 0.0614 | 0.7659 ± 0.1181 |
| PPMI    | **0.7083** ± 0.0572 | 0.6821 ± 0.0860 | 0.6700 ± 0.0587 |

Our method consistently outperformed CNN, GCN, CRGNN, and MGNN baselines.

## Acknowledgments

This work was conducted as part of the **2024 NSF REU** program at UNC Greensboro and supported by **NSF Grant CNS-2349369**.

## Citation

```bibtex
@inproceedings{yang2024edgeboosted,
  title={Edge-Boosted Graph Learning for Functional Brain Connectivity Analysis},
  author={Yang, David and Abdelmegeed, Mostafa and Modl, John and Kim, Minjeong},
  booktitle={IEEE ISBI},
  year={2024}
}
```
