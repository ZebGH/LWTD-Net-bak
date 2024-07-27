# LWTD: Light Weight Transformer-like Dehazing Network
PyTorch implementation of some single image dehazing networks. 

Currently Implemented:


**Prerequisites:**
1. Python 3.7 
2. Pytorch 1.11



## Introduction
- **train.py** and **dehaze.py** are the entry codes for training and testing, respectively.
- **./evaluation.py** contains image quality evaluation metrics, i.e., PSNR and SSIM.
- **./loss.py** contains loss code.
- **./dehaze_video.py** provides scripts for video dehazing. 


## Quick Start
### Train
```
python train.py
```
### Test
```
python dehaze.py
```