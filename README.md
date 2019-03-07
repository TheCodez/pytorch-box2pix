# [WIP] pytorch-box2pix ![alt text](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)

Inofficial PyTorch implementation of [Box2Pix: Single-Shot Instance Segmentation by Assigning Pixels to Object Boxes](https://lmb.informatik.uni-freiburg.de/Publications/2018/UB18) (Uhrig et al., 2018).

## TODO:

Those two are needed to get the project in a state where it can be trained:

- [ ] Priorbox generation
- [ ] mAP metric

Once this is done the instance segmentation part can be added.

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the Cityscapes dataset

## Usage

Train model:

```bash
python train.py --dataset-dir 'data/cityscapes'
```
