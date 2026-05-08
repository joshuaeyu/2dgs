# Image Representation with 2D Gaussians

This repository houses the code I developed for my final project for CS 8803 O27: Computer Graphics in the AI Era for Georgia Tech's OMSCS program. Much of my 2DGS implementation is inspired by the formulation and implementation notes laid out in Zhang et al.'s [Image-GS](https://doi.org/10.1145/3721238.3730596). The main goals of this project were to implement a robust 2D Gaussian-based image representation model and to investigate how image complexity impacts the number of Gaussians needed to achieve a certain level of reconstruction quality.

## Repo Contents

- `train.py` is the main entry point for fitting the 2D Gaussian model to a target image. Run `python train.py -h` for usage. The training loop saves output images every 50 epochs, creates a GIF which animates output images every 10 epochs, plots loss curves, and saves a training log.

- `gs2d.py` contains the core 2DGS methods including Gaussian initialization and rasterization (onto a tensor).

- `metrics.py` contains image similarity and complexity metrics.

- `analyze.ipynb` analyzes the image complexity of a dataset (or a set of datasets), runs the training loop on the entire dataset at varying numbers of Gaussians, and analyzes the resulting relationship between image complexity, number of Gaussians, and reconstruction accuracy. The training step takes a very long time (unless the dataset is small or the list of `n_gaussians` to test is small).

## Excerpts from my Final Report

### Introduction

The main contributions of this project are:

1. Implementation of lightweight 2D Gaussian-based image representation;
2. Demonstration that average image gradient, or edgeness, can be used as an image complexity metric which indicates how well 2D Gaussians can represent a target image;
3. Demonstration that increasing the number of 2D Gaussians improves PSNR independent of image edgeness, but improves structural similarity (SSIM) more for images with high edgeness.

### Conclusion

This project has explored how the performance of 2D Gaussian-based image representation varies based on the complexity of the target image. Using the average image gradient, or edgeness, of images as a complexity metric, it was shown reconstruction accuracy tends to decrease as image complexity increases.

Additionally, the impact of increasing the number of Gaussians was investigated. While PSNR improves with more Gaussians, the maximum possible PSNR seems heavily dependent on image complexity. However, by analyzing SSIM instead, it was found that images with frequent edges benefit most from increasing the number of Gaussians, allowing the 2D Gaussian model to achieve high structural similarity even for more complex images. Based on these findings, an image’s edgeness may be a good indicator of how many 2D Gaussians should be used to reconstruct an image while minimizing computational load.

Future work can utilize more computing power to study a larger dataset, higher- resolution target images, and higher Gaussian counts to improve the generalizability of this project’s findings. Other complexity metrics such as Shannon entropy, angular second moment, contrast, or segmentation can also be explored. Finally, a formal mapping between complexity and required number of Gaussians could be developed.
