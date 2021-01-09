## RTS3D: Real-time Stereo 3D Detection from 4D Feature-Consistency Embedding Space for Autonomous Driving (AAAI2021).

RTS3D is efficiency and accuracy stereo 3D object detection method for autonomous driving.

[**RTS3D**](https://arxiv.org/abs/2012.15072)

## Introduction
RTS3D is the first true real-time system (FPS>24) for stereo image 3D detection meanwhile achieves 10% improvement in average precision comparing with the previous state-of-the-art method.
RTS3D only require RGB images without synthetic data, instance segmentation, CAD model, or depth generator.

## Highlights
- **Fast:** 33 FPS of single image test speed in KITTI benchmark with 384*1280 resolution
- **Accuracy:** SOTA on the KITTI benchmark.
- **Anchor Free:** No 2D or 3D anchor are reauired
- **Easy to deploy:** RTS3D uses conventional convolution operations and MLP, so it is very easy to deploy and accelerate.
## KM3D Baseline and Model Zoo
All experiments are tested with Ubuntu 16.04, Pytorch 1.0.0, CUDA 9.0, Python 3.6, single NVIDIA 2080Ti

IoU Setting 1: Car IoU > 0.5, Pedestrian IoU > 0.25, Cyclist IoU > 0.25

IoU Setting 2: Car IoU > 0.7, Pedestrian IoU > 0.5, Cyclist IoU > 0.5

- Training on KITTI train split and evaluation on val split.
    - FCE Space Resolution: 10 * 10 * 10
    - Model: ([Google Drive](https://drive.google.com/file/d/170B_2Dql8jrbhDRxTXp3vnQzozjPLTEn/view?usp=sharing)), ([Baidu Cloud](https://pan.baidu.com/s/1ZlrwDWQRm_8zGsGIhgOkVw) 提取码：k4uk)

| Class           | Iteration  | FPS  |AP BEV IoU Setting1      | AP 3D IoU Setting1     |AP BEV IoU Setting2      | AP 3D IoU Setting2     |
| :----:          | :----:     | :----:    | :----:                  | :----:                 |:----:                   | :----:                 |
| -               | -          | -         |Easy / Moderate / Hard  | Easy / Moderate / Hard | Easy / Moderate / Hard  | Easy / Moderate / Hard |
| Car- Recall-11  | 1          | 90.9      |89.83, 77.05, 68.28     | 89.27, 70.12, 61.17    | 73.20, 53.62, 46.44     | 60.87, 42.38, 36.44    |
| Car- Recall-40  | 1          | 90.9      |92.92, 76.17, 66.62     | 90.35, 71.37, 63.52    | 78.12, 54.75, 47.09     | 60.34, 39.32, 32.97    |
| Car- Recall-11  | 2          | 45.5      |90.41, 78.70, 70.03     | 90.26, 77.23, 68.28    | 76.56, 56.46, 48.20     | 63.65, 44.50, 37.48    |
| Car- Recall-40  | 2          | 45.5      |95.75, 79.61, 69.69     | 93.57, 76.64, 66.72    | 78.12, 54.75, 47.09     | 63.99, 41.78, 34.96    |



## Installation
Please refer to [INSTALL.md](readme/INSTALL.md)
## Dataset preparation
Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows:
```
KM3DNet
├── kitti_format
│   ├── data
│   │   ├── kitti
│   │   |   ├── annotations
│   │   │   ├── calib /000000.txt .....
│   │   │   ├── image(left[0-7480] right[7481-14961] input augmentatiom)
│   │   │   ├── label /000000.txt .....
|   |   |   ├── train.txt val.txt trainval.txt
│   │   │   ├── mono_results /000000.txt .....
├── src
├── demo_kitti_format
├── readme
├── requirements.txt
```

## Getting Started
Please refer to [GETTING_STARTED.md](readme/GETTING_STARTED.md) to learn more usage about this project.

## Acknowledgement
- [**CenterNet**](https://github.com/xingyizhou/CenterNet)
- [**RTM3D**](https://github.com/Banconxuan/RTM3D)
## License

RTS3D is released under the MIT License (refer to the LICENSE file for details).
Portions of the code are borrowed from, [CenterNet](https://github.com/xingyizhou/CenterNet), [iou3d](https://github.com/sshaoshuai/PointRCNN) and [kitti_eval](https://github.com/prclibo/kitti_eval) (KITTI dataset evaluation). Please refer to the original License of these projects (See [NOTICE](NOTICE)).
## Citation

If you find this project useful for your research, please use the following BibTeX entry.

    @misc{2012.15072,
    Author = {Peixuan Li, Shun Su, Huaici Zhao},
    Title = {RTS3D: Real-time Stereo 3D Detection from 4D Feature-Consistency Embedding Space for Autonomous Driving},
    Year = {2020},
    Eprint = {arXiv:2012.15072},
    }
