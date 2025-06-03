# Deep Stereo Image Inpainting

## Abstract

Immersive VR/AR needs high-quality, consistent stereo images. Single-view inpainting methods ignore stereo geometry, causing visual discomfort. Therefore, stereo consistency in inpainting is essential for realistic and comfortable user experiences. Recent deep learning advances can model relationships between stereo pairs. We adapt the Parallax Attention approach for stereo image inpainting, training from scratch on the 48k pairs of stereo images from Flickr1024 dataset to achieve perceptually coherent results.

## Code Structure

PASSRnet/
├── demo_test.py               # Main demo script
├── demo_test_baseline_L.py    # Demo for left-baseline only model
├── demo_test_baseline_R.py    # Demo for right-baseline only model
├── train.py                   # Main training script
├── train_baseline_L.py        # Train left-view baseline model (no PAM)
├── train_baseline_R.py        # Train right-view baseline model (no PAM)
├── disparity.py               # Disparity estimation and error visualization
├── models.py                  # PASSRnet model, ResB, PAM, etc.
├── utils.py                   # Utilities: PSNR, dataloader, etc.
├── LICENSE
└── README.md

## Reference
```
@InProceedings{Wang2019Learning,
  author    = {Longguang Wang and Yingqian Wang and Zhengfa Liang and Zaiping Lin and Jungang Yang and Wei An and Yulan Guo},
  title     = {Learning Parallax Attention for Stereo Image Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2019},
}
```