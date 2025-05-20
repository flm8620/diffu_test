# DiffuTest: Conditional Diffusion for Next-Digit Image Generation

This repository demonstrates a simple conditional diffusion model that learns to generate an image of digit `n+1` given an image of digit `n`. The dataset is synthetically generated on-the-fly using OpenCV and PIL, and the model is a U-Net with residual and transformer blocks.

## Training

Configure your experiment in `configs/config.yaml` (see [hydra](https://hydra.cc/) for details).

To train:
```bash
python train.py
```

## Results

| Epoch | Input (Digit n) | Target (Digit n+1) | Model Output |
|:-----:|:--------------:|:------------------:|:------------:|
|   1   | ![](images/example_x.png) | ![](images/example_y.png) | ![](images/example_pred.png) |
|   7   | ![](images/epoch7_in.png)   | ![](images/epoch7_gt.png)  | ![](images/epoch7_pred.png)     |
|   15   | ![](images/epoch15_in.png)   | ![](images/epoch15_gt.png)  | ![](images/epoch15_pred.png)     |
|   100   | ![](images/epoch100_in.png)   | ![](images/epoch100_gt.png)  | ![](images/epoch100_pred.png)     |
|   500   | ![](images/epoch500_in.png)   | ![](images/epoch500_gt.png)  | ![](images/epoch500_pred.png)     |

