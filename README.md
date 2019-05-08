<div align="center">
  <img src="https://raw.githubusercontent.com/Z-Zheng/images_repo/master/logo.png"><br><br>
</div>

---------------------
This repo is aimed to simplify training, evaluation and prediction in Pytorch.
## Change logs
- 2019/05/05 compatible with pytorch==1.1.0 (naive sync bn in ddp train)
- 2019/04/08 v0.2.0 released! Support apex!
- 2019/01/25 summary grads and weights
- 2018/12/20 support SE Block
## Features
1. Focus on your research rather than training template codes
2. High performance parallel training using Pytorch 1.1
3. Dynamic module registration mechanism makes you customize components on the fly
4. Support mixed precision training, significantly reducing GPU memory usage with similar performance
5. Support stable distribute training and Sync BN by offical repo and NVIDIA/apex
--------------
## Installation

```bash
pip install --upgrade git+https://github.com/Z-Zheng/simplecv.git
```

#### Requirements:
- pytorch == 1.0.0 or 1.1.0
- tensorboardX
- opencv

## Citing SimpleCV
If you use simplecv in your research, please use the following BibTeX entry.
```
@misc{simplecv2018,
  author =       {Zhuo Zheng},
  title =        {simplecv},
  howpublished = {\url{https://github.com/Z-Zheng/simplecv}},
  year =         {2018}
}
```

## Usage
Please refer to [USAGE.md](https://github.com/Z-Zheng/simplecv/USAGE.md) for the basic usage of simplecv.

## Projects using simplecv
- 2019 IEEE GRSS Data Fusion Contest, Track1: Single-view semantic 3D challenge, 2nd solution code (Pop-Net)


### TODO
- Support more preprocess methods using numpy and pytorch
- [ ] add more preset modules
- [ ] add complete demos of segmentation, detection

### Recent plans
- [ ] add search lr function such as the one in Fast.ai
- [ ] change checkpoint format from model-step.pth to model-step-epoch.pth
- [ ] For 2019 IEEE GRSS Data Fusion Contest, Track1: Single-view semantic 3D challenge, 2nd solution code (Pop-Net) will be released!