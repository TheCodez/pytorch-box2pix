# pytorch-box2pix

Inofficial PyTorch implementation of [Box2Pix: Single-Shot Instance Segmentation by Assigning Pixels to Object Boxes](https://lmb.informatik.uni-freiburg.de/Publications/2018/UB18) (Uhrig et al., 2018).

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the Cityscapes dataset

## Usage

Train model:

```bash
python train.py --lr 0.0001
```

## Usage

```bash
usage: train.py [-h] [--batch_size BATCH_SIZE]
                     [--val_batch_size VAL_BATCH_SIZE]
                     [--epochs EPOCHS] [--lr LR] [--seed SEED]
                     [--output-dir OUTPUT_DIR]

Box2Pix with PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        input batch size for training (default: 16)
  --val_batch_size VAL_BATCH_SIZE
                        input batch size for validation (default: 64)
  --epochs EPOCHS       number of epochs to train (default: 60)
  --lr LR               learning rate (default: 0.0001)
  --seed SEED           manual seed
  --output-dir OUTPUT_DIR
                        directory to save model checkpoints
```
