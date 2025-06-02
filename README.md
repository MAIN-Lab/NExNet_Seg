# NExNet Seg: Neuron Expansion Network for Medical Image Segmentation
We introduce NExNet Seg, the Neuron Expansion Network for Medical Image Segmentation. Inspired by Progressively Expanded Neu- ron (PEN) structures and Manhattan Self-Attention (MaSA) mechanisms, NExNet Seg achieves exceptional accuracy with high parameter efficiency.

## Overview
NExNet Seg is a novel deep learning architecture designed for efficient and accurate medical image segmentation, as presented in the LXCV-CVPRw 2025 submission (Paper ID 3). It integrates **Trainable Progressively Expanded Neurons (T-PEN)**, **Manhattan Self-Attention (MaSA)**, and **Self-Supervised Learning (SSL)** to achieve superior performance with reduced computational cost. The model excels in tasks such as skin lesion segmentation (e.g., ISIC datasets) and gastrointestinal polyp segmentation (e.g., Kvasir-Seg, CVC-Clinic DB), balancing high accuracy with computational efficiency for clinical deployment.

This repository contains the implementation of NExNet Seg, including training scripts and configurations for reproducing the results reported in the paper.

## Key Features
- **T-PEN**: Enhances feature representation by expanding neurons with trainable coefficients, improving adaptability over the original PEN framework.
- **MaSA**: Utilizes Manhattan Self-Attention to capture spatially relevant features efficiently, reducing computational overhead.
- **SSL**: Leverages self-supervised pretraining with a U-Net-based denoising model to improve generalization.
- **Efficiency**: Achieves high Dice Coefficients (e.g., 0.943 on Kvasir-Seg) with only 30.4M parameters, compared to TransUNet's 105M.
- **Applications**: Demonstrates robust performance on skin lesion (ISIC 2016-2018, PH²) and polyp segmentation (Kvasir-Seg, CVC-Clinic DB).

## Architecture
The NExNet Seg architecture combines T-PEN and MaSA within a U-Net-like encoder-decoder framework. The encoder uses Neuron Expansion (NEx) blocks with T-PEN to enrich feature representations, while MaSA enhances spatial feature extraction in Conv + MaSA blocks. Skip connections integrate MaSA features into the decoder for accurate segmentation mask reconstruction.

<p align="center">
  <figure>
    <img width="750" src="images/NExNetSeg_detailed.png" alt="NexNetSeg Architecture">
    <figcaption>Illustration of the proposed NExNet Seg architecture for medical image segmentation. (a) NExNet Seg combines T-PEN and MaSA concepts. (b) T-PEN is embedded into the NEx blocks, while MaSA is combined with traditional convolution within the Conv MaSA block.</figcaption>
  </figure>
</p>

## Installation
### Prerequisites
- Python 3.8+
- PyTorch with CUDA support
- NVIDIA GPU (e.g., A-100 with 40GB for training)
- Anaconda for environment management
- Datasets: ISIC 2016-2018, PH², Kvasir-Seg, CVC-Clinic DB

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MAIN-Lab/NExNet_Seg.git
   cd NExNet_Seg
   ```

2. **Create Conda Environment**:
   ```bash
   conda create -n upen_torch python=3.8
   conda activate upen_torch
   ```

3. **Install Dependencies**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122
   pip install -r requirements.txt
   ```

## Usage
### Training
The repository includes scripts for training the autoencoder (SSL pretraining) and the NExNet Seg model. Example commands are provided below.

#### Self-Supervised Pretraining
- For skin lesion datasets:
  ```bash
  sbatch auto_ssl_skin.sh
  ```
- For polyp datasets:
  ```bash
  sbatch auto_ssl_colon.sh
  ```

#### NExNet Seg Training
- For polyp segmentation (e.g., Kvasir-Seg):
  ```bash
  sbatch nexnet_train_colon.sh
  ```

Modify the scripts to adjust hyperparameters such as `--num_epochs`, `--batch_size`, `--dropout_rate`, or `--gamma` as needed.

### Evaluation
Evaluate the model using the provided datasets and metrics (Dice Coefficient, IoU). The trained models can be tested with:
```bash
python evaluate.py --model_path <path_to_trained_model> --dataset <dataset_name>
```

## Results
NExNet Seg outperforms several state-of-the-art models in medical image segmentation, as shown in the table below (extracted from Table 1 in the paper).

| Method         | ISIC 16 (DC) | ISIC 17 (DC) | ISIC 18 (DC) | PH² (DC) | Kvasir-Seg (DC) | CVC-Clinic (DC) |
|----------------|--------------|--------------|--------------|----------|-----------------|-----------------|
| UNet           | 0.887        | 0.794        | 0.813        | 0.873    | 0.775           | 0.856           |
| UNet++         | 0.889        | 0.814        | 0.833        | 0.890    | 0.786           | 0.874           |
| TransUNet      | 0.913        | 0.873        | 0.904        | 0.910    | 0.889           | 0.920           |
| SegFormer      | -            | 0.851        | 0.869        | -        | 0.905           | 0.931           |
| NExNet Seg     | **0.913**    | **0.921**    | **0.918**    | **0.936**| **0.943**       | **0.934**       |

![Qualitative Comparison](figures/figure7.png)
*Figure 7: Qualitative comparison of NExNet Seg with state-of-the-art methods. Blue contours represent ground truth, green contours represent predictions.*

### Ablation Study
The ablation study (Table 2) demonstrates the contribution of each component (T-PEN, MaSA, SSL). The full model with all components achieves the highest performance across datasets.

![Ablation Study](figures/figure8b.png)
*Detailed examination of the proposed methodology. (a) Illustrates the performance capability of NExNet Seg with fewer
trainable parameters in comparison with the state-of-the-art. [Different markers represent compared models, and different colors represent
the dataset used in experiments] (b) Ablation analysis performance comparison.*

## Computational Efficiency
NExNet Seg is designed for resource-constrained environments, with fewer parameters and lower inference times compared to models like TransUNet (Table 4).

| Method        | Parameters (M) | Inference Time (s) | FLOPS (G) |
|---------------|----------------|--------------------|-----------|
| UNet          | 31.1           | 0.09               | 68        |
| TransUNet     | 105            | 0.30               | 160       |
| NExNet Seg    | **30.4**       | **0.06**           | **18**    |

## Limitations
- **Overfitting**: T-PEN layers may overfit on certain datasets (e.g., ISIC17, CVC-Clinic).
- **Generalization**: Performance on other medical imaging tasks is untested.
- **Hardware**: Training requires high-end GPUs (e.g., NVIDIA A-100).

## Future Work
- Explore regularization techniques to mitigate T-PEN overfitting.
- Test NExNet Seg on additional medical imaging tasks.
- Optimize for lower-end hardware to improve accessibility.





   
# UNDER CONSTRUCTION
### In the meantime, here is a picture of a 
<p align="center">
  <figure>
    <img width="700" src="images/quokka.jpg" alt="quokka">
    <figcaption>Happy quokka.</figcaption>
  </figure>
</p>

## License
See the [LICENSE](LICENSE.md) file for license rights and limitations (MIT).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
