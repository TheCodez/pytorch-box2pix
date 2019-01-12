import warnings
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from ignite.utils import convert_tensor
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, Engine, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Accuracy, Loss, RunningAverage
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.cityscapes import CityscapesDataset
from datasets.transforms.transforms import Resize, Compose, RandomHorizontalFlip, ToTensor
from loss.box2pixloss import box2pix_loss, Box2PixLoss
from loss.multiboxloss import MultiBoxLoss
from models.box2pix import Box2Pix


def get_data_loaders(train_batch_size, val_batch_size):
    new_size = (512, 1024)  # (1024, 2048)

    joint_transforms = Compose([
        Resize(new_size),
        RandomHorizontalFlip(),
        ToTensor()
    ])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = DataLoader(CityscapesDataset(root='data/cityscapes', split='train', joint_transform=joint_transforms,
                                                img_transform=normalize),
                              batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(CityscapesDataset(root='data/cityscapes', split='val', joint_transform=ToTensor(),
                                              img_transform=normalize),
                            batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


def run(train_batch_size, val_batch_size, epochs, lr, seed, output_dir):
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)

    if seed is not None:
        torch.manual_seed(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Box2Pix()

    if torch.cuda.device_count() > 1:
        print('Using %d GPU(s)' % torch.cuda.device_count())
        model = nn.DataParallel(model)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    semantics_criterion = nn.CrossEntropyLoss(ignore_index=255)
    offsets_criterion = nn.MSELoss()
    box_criterion = MultiBoxLoss()
    multitask_criterion = Box2PixLoss()

    def _prepare_batch(batch, dev=None, non_blocking=True):
        x, instance, boxes, confs = batch

        return (convert_tensor(x, device=dev, non_blocking=non_blocking),
                convert_tensor(instance, device=dev, non_blocking=non_blocking),
                convert_tensor(boxes, device=dev, non_blocking=non_blocking),
                convert_tensor(confs, device=dev, non_blocking=non_blocking))

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, instance, boxes, confs = _prepare_batch(batch, dev=device)

        loc_preds, conf_preds, semantics_pred, offsets_pred = model(x)

        semantics_loss = semantics_criterion(semantics_pred, instance)
        offsets_loss = offsets_criterion(offsets_pred, instance)
        box_loss, conf_loss = box_criterion(loc_preds, conf_preds, boxes, confs)

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

    checkpoint_handler = ModelCheckpoint(output_dir, 'checkpoint', save_interval=1, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    # attach running average metrics
    monitoring_metrics = ['loss', 'loss_semantics', 'loss_offsets', 'loss_ssdbox', 'loss_ssdclass']
    RunningAverage(alpha=0.98, output_transform=lambda x: x['loss']).attach(trainer, 'loss')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['loss_semantics']).attach(trainer, 'loss_semantics')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['loss_offsets']).attach(trainer, 'loss_offsets')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['loss_ssdbox']).attach(trainer, 'loss_ssdbox')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['loss_ssdclass']).attach(trainer, 'loss_ssdclass')

    # attach progress bar
    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler, to_save={'model': model})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'loss_semantics': Loss(semantics_criterion),
                                                     'loss_offsets': Loss(offsets_criterion),
                                                     'loss_box': Loss(box_criterion)},
                                            device=device)

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        pbar.log_message('Epoch [{}/{}] done. Time per batch: {:.3f}[s]'.format(engine.state.epoch,
                                                                                engine.state.max_epochs, timer.value()))
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        loss_semantics = metrics['loss_semantics']
        loss_offsets = metrics['loss_offsets']
        loss_ssdbox = metrics['loss_ssdbox']
        loss_ssdclass = metrics['loss_ssdclass']
        pbar.log_message('Training Results - Epoch: [{}/{}]  Avg accuracy: {:.4f} Avg semantic loss: {:.4f} '
                         'Avg offsets loss: {:.4f} Avg ssdbox loss: {:.4f} Avg ssdclass loss: {:.4f}'
                         .format(engine.state.epoch, engine.state.max_epochs, avg_accuracy, loss_semantics,
                                 loss_offsets, loss_ssdbox, loss_ssdclass))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        loss_semantics = metrics['loss_semantics']
        loss_offsets = metrics['loss_offsets']
        loss_ssdbox = metrics['loss_ssdbox']
        loss_ssdclass = metrics['loss_ssdclass']
        pbar.log_message('Validation Results - Epoch: [{}/{}]  Avg accuracy: {:.4f} Avg semantic loss: {:.4f} '
                         'Avg offsets loss: {:.4f} Avg ssdbox loss: {:.4f} Avg ssdclass loss: {:.4f}'
                         .format(engine.state.epoch, engine.state.max_epochs, avg_accuracy, loss_semantics,
                                 loss_offsets, loss_ssdbox, loss_ssdclass))

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            engine.terminate()
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')

            checkpoint_handler(engine, {'model_exception': model})

        else:
            raise e

    trainer.run(train_loader, max_epochs=epochs)


if __name__ == '__main__':
    parser = ArgumentParser('Box2Pix with PyTorch')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--val_batch_size', type=int, default=64,
                        help='input batch size for validation (default: 64)')
    parser.add_argument('--epochs', type=int, default=60,
                        help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int,
                        help='manual seed')
    parser.add_argument('--output-dir', default='./checkpoints',
                        help='directory to save model checkpoints')

    args = parser.parse_args()

    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.seed, args.output_dir)
