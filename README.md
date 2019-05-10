# [WIP] pytorch-box2pix ![alt text](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)

Inofficial PyTorch implementation of [Box2Pix: Single-Shot Instance Segmentation by Assigning Pixels to Object Boxes](https://lmb.informatik.uni-freiburg.de/Publications/2018/UB18) (Uhrig et al., 2018).

## TODO:

This is needed to get the project in a state where it can be trained:

- [ ] mAP metric

Instance segmentation can be added later as it's just a post processing step.

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt` (Currently requires torchvision master)
- Download the Cityscapes dataset

## Usage

Train model:

```bash
python train.py --dataset-dir 'data/cityscapes'
```
