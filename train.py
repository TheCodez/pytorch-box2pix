import os
import random
from argparse import ArgumentParser
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Events, Engine
from ignite.metrics import RunningAverage, Loss, ConfusionMatrix, mIoU
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

from datasets.cityscapes import CityscapesDataset
from datasets.transforms import transforms
from loss.boxloss import BoxLoss
from loss.multitaskloss import MultiTaskLoss
from metrics.mean_ap import MeanAveragePrecision
from model import Box2Pix
from utils.box_coder import BoxCoder
from utils.helper import save


def get_data_loaders(data_dir, batch_size, num_workers):
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.RandomGaussionBlur(radius=2.0),
        transforms.ToTensor(),
        transforms.ConvertIdToTrainId(),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_loader = DataLoader(CityscapesDataset(root=data_dir, split='train', transforms=transform),
                              batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(CityscapesDataset(root=data_dir, split='val', transforms=val_transform),
                            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def run(args):
    train_loader, val_loader = get_data_loaders(args.dataset_dir, args.batch_size, args.num_workers)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = CityscapesDataset.num_instance_classes() + 1
    model = Box2Pix(num_classes, init_googlenet=True)

    if torch.cuda.device_count() > 1:
        print("Using %d GPU(s)" % torch.cuda.device_count())
        model = nn.DataParallel(model)

    model = model.to(device)

    semantics_criterion = nn.CrossEntropyLoss(ignore_index=255)
    offsets_criterion = nn.MSELoss()
    box_criterion = BoxLoss(num_classes, gamma=2)
    multitask_criterion = MultiTaskLoss().to(device)

    box_coder = BoxCoder()
    optimizer = optim.Adam([{'params': model.parameters()},
                            {'params': multitask_criterion.parameters()}], lr=args.lr)

    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            args.start_iteration = checkpoint['iteration']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            multitask_criterion.load_state_dict(checkpoint['multitask'])
            print("Loaded checkpoint '{}' (Epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    def _prepare_batch(batch, non_blocking=True):
        image, instance, centroids, boxes, labels = batch

        return (convert_tensor(image, device=device, non_blocking=non_blocking),
                convert_tensor(instance, device=device, non_blocking=non_blocking),
                convert_tensor(centroids, device=device, non_blocking=non_blocking),
                convert_tensor(boxes, device=device, non_blocking=non_blocking),
                convert_tensor(labels, device=device, non_blocking=non_blocking))

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        image, instance, centroids, boxes, labels = _prepare_batch(batch)
        boxes, labels = box_coder.encode(boxes, labels)

        loc_preds, conf_preds, semantics_pred, offsets_pred = model(image)

        semantics_loss = semantics_criterion(semantics_pred, instance)
        offsets_loss = offsets_criterion(offsets_pred, centroids)
        box_loss, conf_loss = box_criterion(loc_preds, boxes, conf_preds, labels)

        loss = multitask_criterion(semantics_loss, offsets_loss, box_loss, conf_loss)

        loss.backward()
        optimizer.step()

        return {
            'loss': loss.item(),
            'loss_semantics': semantics_loss.item(),
            'loss_offsets': offsets_loss.item(),
            'loss_ssdbox': box_loss.item(),
            'loss_ssdclass': conf_loss.item()
        }

    trainer = Engine(_update)

    # attach running average metrics
    train_metrics = ['loss', 'loss_semantics', 'loss_offsets', 'loss_ssdbox', 'loss_ssdclass']
    for m in train_metrics:
        transform = partial(lambda x, metric: x[metric], metric=m)
        RunningAverage(output_transform=transform).attach(trainer, m)

    # attach progress bar
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=train_metrics)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            image, instance, boxes, labels = _prepare_batch(batch)
            loc_preds, conf_preds, semantics_pred, offsets_pred = model(image)
            boxes_preds, labels_preds, scores_preds = box_coder.decode(loc_preds, F.softmax(conf_preds, dim=1),
                                                                       score_thresh=0.01)

            semantics_loss = semantics_criterion(semantics_pred, instance)
            offsets_loss = offsets_criterion(offsets_pred, instance)
            box_loss, conf_loss = box_criterion(loc_preds, boxes, conf_preds, labels)

            instances_pred = box_coder.assign_box2pix(semantics_pred, offsets_pred, boxes_preds, labels_preds)

            return {
                'loss': (semantics_loss, offsets_loss, {'box_loss': box_loss, 'conf_loss': conf_loss}),
                'objects': (boxes_preds, labels_preds, scores_preds, boxes, labels),
                'semantics': (semantics_pred, instance),
                'instances': instances_pred
            }

    evaluator = Engine(_inference)
    cm = ConfusionMatrix(num_classes, output_transform=lambda x: x['semantics'])
    mIoU(cm).attach(evaluator, 'mIoU')
    Loss(multitask_criterion, output_transform=lambda x: x['loss']).attach(evaluator, 'loss')
    MeanAveragePrecision(num_classes, output_transform=lambda x: x['objects']).attach(evaluator, 'mAP')

    pbar2 = ProgressBar(persist=True, desc='Eval Epoch')
    pbar2.attach(evaluator)

    def _global_step_transform(engine, event_name):
        return trainer.state.iteration

    exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_logger = TensorboardLogger(os.path.join(args.log_dir, exp_name))
    tb_logger.attach(trainer,
                     log_handler=OutputHandler(tag='training', output_transform=lambda out: {
                         'loss': out['loss'],
                         'loss_semantics': out['loss_semantics'],
                         'loss_offsets': out['loss_offsets'],
                         'loss_ssdbox': out['loss_ssdbox'],
                         'loss_ssdclass': out['loss_ssdclass']

                     }),
                     event_name=Events.ITERATION_COMPLETED)

    tb_logger.attach(evaluator,
                     log_handler=OutputHandler(tag='validation',
                                               metric_names=['loss', 'mAP', 'mIoU'],
                                               output_transform=lambda out: {
                                                   'loss': out['loss'],
                                                   'objects': out['objects'],
                                                   'semantics': out['semantics']
                                               },
                                               global_step_transform=_global_step_transform),
                     event_name=Events.EPOCH_COMPLETED)

    @trainer.on(Events.STARTED)
    def initialize(engine):
        if args.resume:
            engine.state.epoch = args.start_epoch
            engine.state.iteration = args.start_iteration

    @evaluator.on(Events.EPOCH_COMPLETED)
    def save_checkpoint(engine):
        mean_iou = engine.state.metrics['mIoU'] * 100.0

        name = 'epoch{}_mIoU={:.1f}.pth'.format(trainer.state.epoch, mean_iou)
        file = {'model': model.state_dict(), 'epoch': trainer.state.epoch, 'iteration': engine.state.iteration,
                'optimizer': optimizer.state_dict(), 'args': args, 'bestIoU': trainer.state.best_iou}

        save(file, args.output_dir, 'checkpoint_{}'.format(name))
        save(model.state_dict(), args.output_dir, 'model_{}'.format(name))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        pbar.log_message("Start Validation - Epoch: [{}/{}]".format(engine.state.epoch, engine.state.max_epochs))
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        loss = metrics['loss']
        mean_ap = metrics['mAP']
        iou = metrics['mIoU']

        pbar.log_message("Validation results - Epoch: [{}/{}]: Loss: {:.2e}, mAP(50%): {:.1f}, IoU: {:.1f}"
                         .format(engine.state.epoch, engine.state.max_epochs, loss, mean_ap * 100.0, iou * 100.0))

    print("Start training")
    trainer.run(train_loader, max_epochs=args.epochs)
    tb_logger.close()


if __name__ == '__main__':
    parser = ArgumentParser('Box2Pix with PyTorch')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='input batch size for training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=123,
                        help='manual seed')
    parser.add_argument('--output-dir', default='checkpoints',
                        help='directory to save model checkpoints')
    parser.add_argument('--resume', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='log directory for Tensorboard log output')
    parser.add_argument('--dataset-dir', type=str, default='data/cityscapes',
                        help='location of the dataset')

    run(parser.parse_args())
