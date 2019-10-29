<div align="center">
  <img src="https://raw.githubusercontent.com/Z-Zheng/images_repo/master/logo.png"><br><br>
</div>

---------------------
This repo is aimed to simplify training, evaluation and prediction in Pytorch.
## Change logs
- 2019/10/29 v0.3.4 released! More preset models have been added.
- 2019/06/25 v0.3.1 released! More features have been added.
- 2019/05/24 v0.3.0 released! 
- 2019/05/05 compatible with pytorch==1.1.0 (naive sync bn in ddp train)
- 2019/04/08 v0.2.0 released! Support apex!
- 2019/01/25 summary grads and weights
- 2018/12/20 support SE Block
## Features
1. Focus on your research rather than training template codes
2. Dynamic module registration mechanism makes you customize components on the fly
3. Flexible plugin mechanism for a hackable trainer without any coupling!
4. High performance parallel training using Pytorch 1.1
5. Support mixed precision training, significantly reducing GPU memory usage with similar performance
6. Support stable distribute training and Sync BN by offical repo and NVIDIA/apex
--------------
## Installation

```bash
pip install --upgrade git+https://github.com/Z-Zheng/SimpleCV.git
```

#### Requirements:
- pytorch == 1.1.0
- tensorboardX
- opencv

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
Please refer to [USAGE.md](https://github.com/Z-Zheng/simplecv/USAGE.md) for the basic usage of SimpleCV.

## Projects using SimpleCV
- 2019 IEEE GRSS Data Fusion Contest, Track1: Single-view semantic 3D challenge, 2nd solution code (Pop-Net)


### TODO
- Support more preprocess methods using numpy and pytorch
- [ ] add detailed API doc and tutorial
- [ ] add more preset modules
- [ ] add complete demos of segmentation, detection

### Recent plans
- [ ] For 2019 IEEE GRSS Data Fusion Contest, Track1: Single-view semantic 3D challenge, 2nd solution code (Pop-Net) will be released!