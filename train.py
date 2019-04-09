import os
import warnings
from argparse import ArgumentParser
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Loss, ConfusionMatrix, mIoU
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

import models
from datasets.cityscapes import CityscapesDataset
from datasets.transforms import transforms
from datasets.transforms.transforms import ToTensor
from loss.boxloss import BoxLoss
from loss.multitaskloss import MultiTaskLoss
from metrics.mean_ap import MeanAveragePrecision
from utils import helper
from utils.box_coder import BoxCoder


def get_data_loaders(data_dir, batch_size, num_workers):
    # new_size = (512, 1024)  # (1024, 2048)

    joint_transform = transforms.Compose([
        # transforms.Resize(new_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomGaussionBlur(sigma=(0, 1.2)),
        transforms.ToTensor()
    ])

    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = DataLoader(CityscapesDataset(root=data_dir, split='train', joint_transform=joint_transform,
                                                img_transform=normalize),
                              batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(CityscapesDataset(root=data_dir, split='val', joint_transform=ToTensor(),
                                              img_transform=normalize),
                            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def run(args):
    train_loader, val_loader = get_data_loaders(args.dir, args.batch_size, args.num_workers)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = CityscapesDataset.num_instance_classes() + 1
    model = models.box2pix(num_classes)
    model.init_from_googlenet()

    if torch.cuda.device_count() > 1:
        print("Using %d GPU(s)" % torch.cuda.device_count())
        model = nn.DataParallel(model)

    model = model.to(device)

    semantics_criterion = nn.CrossEntropyLoss(ignore_index=255)
    offsets_criterion = nn.MSELoss()
    box_criterion = BoxLoss(num_classes, gamma=2)
    multitask_criterion = MultiTaskLoss().to(device)

    box_coder = BoxCoder()
    optimizer = optim.Adam([{'params': model.parameters(), 'weight_decay': 5e-4},
                            {'params': multitask_criterion.parameters()}], lr=args.lr)

    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            multitask_criterion.load_state_dict(checkpoint['multitask'])
            print("Loaded checkpoint '{}' (Epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    def _prepare_batch(batch, non_blocking=True):
        x, instance, boxes, labels = batch

        return (convert_tensor(x, device=device, non_blocking=non_blocking),
                convert_tensor(instance, device=device, non_blocking=non_blocking),
                convert_tensor(boxes, device=device, non_blocking=non_blocking),
                convert_tensor(labels, device=device, non_blocking=non_blocking))

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, instance, boxes, labels = _prepare_batch(batch)
        boxes, labels = box_coder.encode(boxes, labels)

        loc_preds, conf_preds, semantics_pred, offsets_pred = model(x)

        semantics_loss = semantics_criterion(semantics_pred, instance)
        offsets_loss = offsets_criterion(offsets_pred, instance)
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

    checkpoint_handler = ModelCheckpoint(args.output_dir, 'checkpoint', save_interval=1, n_saved=10,
                                         require_empty=False, create_dir=True, save_as_state_dict=False)
    timer = Timer(average=True)

    # attach running average metrics
    train_metrics = ['loss', 'loss_semantics', 'loss_offsets', 'loss_ssdbox', 'loss_ssdclass']
    for m in train_metrics:
        transform = partial(lambda x, metric: x[metric], metric=m)
        RunningAverage(output_transform=transform).attach(trainer, m)

    # attach progress bar
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=train_metrics)

    checkpoint = {'model': model.state_dict(), 'epoch': trainer.state.epoch, 'optimizer': optimizer.state_dict(),
                  'multitask': multitask_criterion.state_dict()}
    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler, to_save={
        'checkpoint': checkpoint
    })

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, instance, boxes, labels = _prepare_batch(batch)
            loc_preds, conf_preds, semantics, offsets_pred = model(x)
            boxes_preds, labels_preds, scores_preds = box_coder.decode(loc_preds, F.softmax(conf_preds, dim=1),
                                                                       score_thresh=0.01)

            semantics_loss = semantics_criterion(semantics, instance)
            offsets_loss = offsets_criterion(offsets_pred, instance)
            box_loss, conf_loss = box_criterion(loc_preds, boxes, conf_preds, labels)

            semantics_pred = semantics.argmax(dim=1)
            instances = helper.assign_pix2box(semantics_pred, offsets_pred, boxes_preds, labels_preds)

        return {
            'loss': (semantics_loss, offsets_loss, {'box_loss': box_loss, 'conf_loss': conf_loss}),
            'objects': (boxes_preds, labels_preds, scores_preds, boxes, labels),
            'semantics': semantics_pred,
            'instances': instances
        }

    train_evaluator = Engine(_inference)
    cm = ConfusionMatrix(num_classes=num_classes, output_transform=lambda x: x['semantics'])
    mIoU(cm, ignore_index=0).attach(train_evaluator, 'mIoU')
    Loss(multitask_criterion, output_transform=lambda x: x['loss']).attach(train_evaluator, 'loss')
    MeanAveragePrecision(num_classes, output_transform=lambda x: x['objects']).attach(train_evaluator, 'mAP')

    evaluator = Engine(_inference)
    cm2 = ConfusionMatrix(num_classes=num_classes, output_transform=lambda x: x['semantics'])
    mIoU(cm2, ignore_index=0).attach(train_evaluator, 'mIoU')
    Loss(multitask_criterion, output_transform=lambda x: x['loss']).attach(evaluator, 'loss')
    MeanAveragePrecision(num_classes, output_transform=lambda x: x['objects']).attach(evaluator, 'mAP')

    tb_logger = TensorboardLogger(args.log_dir)
    tb_logger.attach(trainer,
                     log_handler=OutputHandler(tag='training', output_transform=lambda loss: {
                         'loss': loss['loss'],
                         'loss_semantics': loss['loss_semantics'],
                         'loss_offsets': loss['loss_offsets'],
                         'loss_ssdbox': loss['loss_ssdbox'],
                         'loss_ssdclass': loss['loss_ssdclass']

                     }),
                     event_name=Events.ITERATION_COMPLETED)

    tb_logger.attach(train_evaluator,
                     log_handler=OutputHandler(tag='training_eval',
                                               metric_names=['loss', 'mAP', 'mIoU'],
                                               output_transform=lambda loss: {
                                                   'loss': loss['loss'],
                                                   'objects': loss['objects'],
                                                   'semantics': loss['semantics']
                                               },
                                               another_engine=trainer),
                     event_name=Events.EPOCH_COMPLETED)

    tb_logger.attach(evaluator,
                     log_handler=OutputHandler(tag='validation_eval',
                                               metric_names=['loss', 'mAP', 'mIoU'],
                                               output_transform=lambda loss: {
                                                   'loss': loss['loss'],
                                                   'objects': loss['objects'],
                                                   'semantics': loss['semantics']
                                               },
                                               another_engine=trainer),
                     event_name=Events.EPOCH_COMPLETED)

    @trainer.on(Events.STARTED)
    def initialize(engine):
        if args.resume:
            engine.state.epoch = args.start_epoch

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        pbar.log_message("Epoch [{}/{}] done. Time per batch: {:.3f}[s]".format(engine.state.epoch,
                                                                                engine.state.max_epochs, timer.value()))
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        loss = metrics['loss']
        mean_ap = metrics['mAP']
        iou = metrics['mIoU']

        pbar.log_message('Training results - Epoch: [{}/{}]: Loss: {:.4f}, mAP(50%): {:.1f}, IoU: {:.1f}'
                         .format(loss, evaluator.state.epochs, evaluator.state.max_epochs, mean_ap, iou * 100.0))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        loss = metrics['loss']
        mean_ap = metrics['mAP']
        iou = metrics['mIoU']

        pbar.log_message('Validation results - Epoch: [{}/{}]: Loss: {:.4f}, mAP(50%): {:.1f}, IoU: {:.1f}'
                         .format(loss, evaluator.state.epochs, evaluator.state.max_epochs, mean_ap, iou * 100.0))

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            engine.terminate()
            warnings.warn("KeyboardInterrupt caught. Exiting gracefully.")

            checkpoint_handler(engine, {'model_exception': model})
        else:
            raise e

    @trainer.on(Events.COMPLETED)
    def save_final_model(engine):
        checkpoint_handler(engine, {'final': model})

    trainer.run(train_loader, max_epochs=args.epochs)
    tb_logger.close()


if __name__ == '__main__':
    parser = ArgumentParser('Box2Pix with PyTorch')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--seed', type=int, help='manual seed')
    parser.add_argument('--output-dir', default='./checkpoints',
                        help='directory to save model checkpoints')
    parser.add_argument('--resume', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log-dir", type=str, default="tensorboard_logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument("--dataset-dir", type=str, default="data/cityscapes",
                        help="location of the dataset")

    run(parser.parse_args())
