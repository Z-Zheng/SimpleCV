[![DOI](https://zenodo.org/badge/161282115.svg)](https://zenodo.org/badge/latestdoi/161282115)
<div align="center">
  <img src="https://raw.githubusercontent.com/Z-Zheng/images_repo/master/logo.png"><br><br>
</div>

---------------------
This repo is aimed to simplify training, evaluation and prediction in Pytorch.

## Features
1. Focus on your research rather than training template codes
2. Dynamic module registration mechanism makes you customize components on the fly
3. Flexible plugin mechanism for a hackable trainer without any coupling!
4. High performance parallel training using Pytorch
5. Support mixed precision training, significantly reducing GPU memory usage with similar performance
6. Support stable distribute training and Sync BN by offical repo and NVIDIA/apex
--------------
## Installation

```bash
pip install --upgrade git+https://github.com/Z-Zheng/SimpleCV.git
```

#### Requirements:
- pytorch >= 1.1.0
- tensorboardX
- opencv, skimage, sklearn, pillow


## Citing SimpleCV
If you use SimpleCV in your research, please use the following BibTeX entry.
```
@misc{simplecv2018,
  author =       {Zhuo Zheng},
  title =        {SimpleCV},
  howpublished = {\url{https://github.com/Z-Zheng/SimpleCV}},
  year =         {2018}
}
```

## Usage
Please refer to [USAGE.md](https://github.com/Z-Zheng/SimpleCV/blob/master/USAGE.md) for the basic usage of SimpleCV.

## Projects using SimpleCV
- 2019 IEEE GRSS Data Fusion Contest, Track1: Single-view semantic 3D challenge, 2nd solution code (Pop-Net)
- 2020 xView2 Assess Building Damage Challenge, 4th solution code
- Official implementation of our work: [Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zheng_Foreground-Aware_Relation_Network_for_Geospatial_Object_Segmentation_in_High_Spatial_CVPR_2020_paper.pdf)
- Official implementation of our work: [FPGA: Fast Patch-Free Global Learning Framework for Fully End-to-End Hyperspectral Image Classification, TGRS 2020](https://ieeexplore.ieee.org/document/9007624)
## Change logs
- 2019/10/29 v0.3.4 released! More preset models have been added.
- 2019/06/25 v0.3.1 released! More features have been added.
- 2019/05/24 v0.3.0 released! 
- 2019/05/05 compatible with pytorch==1.1.0 (naive sync bn in ddp train)
- 2019/04/08 v0.2.0 released! Support apex!
- 2019/01/25 summary grads and weights
- 2018/12/20 support SE Block