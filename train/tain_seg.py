import os
import csv
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import torch
from torch import optim
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import CyclicLR

from models.seg.UNet.unet_model import UNet, R2UNet
from models.seg.MaskRCNN.model import MaskRCNNResNet50

from utils.indicators import segmentation_indicators, detection_indicators
from utils.utils import EarlyStopping
from utils.data_loading import SegmentDatasetJson

annotation_path = '../datasets/json/maskrcnn_annotation_HAM10000.json'
image_dir = 'E:/Datas/work/HairEffect/RawData/HAM10000/HAM10000_images'


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images_0 and target masks_f')
    parser.add_argument('--segmentation-task', type=bool, dest='st',
                        default=True, help='Choose whether is a segmentation task')
    parser.add_argument('--detection-task', type=bool, dest='dt',
                        default=True, help='Choose whether is a detection task')
    parser.add_argument('--image_dir', '-i', type=str,
                        default=image_dir, help="input image directory")
    parser.add_argument('--json_dir', '-j', type=str,
                        default=annotation_path, help="input json directory")
    parser.add_argument('--model_name', '-m', type=str,
                        default='MASKRCNN', help="select train model_name")
    parser.add_argument('--cycle-times', '-t', type=int,
                        default=4, help='cycle times in R2UNet')
    parser.add_argument('--epochs', '-e', metavar='E', type=int,
                        default=200, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int,
                        default=4, help='Batch size')
    parser.add_argument('--gradient-clipping', dest='gc', type=float,
                        default=5.0, help='Upper limit of training gradient norm')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float,
                        default=1e-5, help='Learning rate', dest='lr')
    parser.add_argument('--size', '-s', type=float,
                        default=(256, 256), help='Downscaling factor of the images_0')
    parser.add_argument('--cuda', type=bool,
                        default=True, help='True means use cuda')
    parser.add_argument('--validation', '-v', dest='val', type=float,
                        default=20, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true',
                        default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true',
                        default=False, help='Use bilinear upSampling')
    parser.add_argument('--n_classes', '-c', type=int,
                        default=2, help='Number of n_classes')

    return parser.parse_args()


def merge_masks(masks):
    """
    Merge masks

    (torch.Tensor): Mask tensor with shape [N, 1, H, W], where N is the number of detected objects.
    threshold (float): Barbarization threshold
.
    Returns
        torch.Tensor: The merged mask has shape [1, H, W].
    """
    if masks.size(0) == 0:
        # Handle the case of empty tensors, such as returning an empty mask
        return torch.zeros((1, masks.size(2), masks.size(3)))
    combined_mask = torch.max(masks, dim=0)[0]  # Take the maximum merge mask
    return combined_mask


class Segmentation_Trainer:
    def __init__(self):
        args = get_args()

        # get parser args
        self.segmentation_task = args.st
        self.detection_task = args.dt
        self.image_dir = args.image_dir
        self.json_dir = args.json_dir
        self.model_name = args.model_name.upper()
        self.cycle_times = args.cycle_times
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.gradient_clipping = args.gc
        self.learning_rate = args.lr
        self.size = args.size
        self.cuda = args.cuda
        self.validation = args.val
        self.amp = args.amp
        self.bilinear = args.bilinear
        self.n_classes = args.n_classes

        # initial args
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')  # set logging
        self.device = torch.device('cuda' if torch.cuda.is_available() & self.cuda else 'cpu')  # set device
        assert self.segmentation_task or self.detection_task, "Please choose the segmentation or detection or both of tasks"

        # set checkpoint and indicators save place
        now_time, now_h = datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%H")
        self.checkpoint_save_dir = Path(f'./checkpoints/{self.model_name}/{now_time}/{now_h}')
        self.indicators_save_dir = Path(f'./indicators/{self.model_name}/{now_time}/{now_h}')
        Path(self.checkpoint_save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.indicators_save_dir).mkdir(parents=True, exist_ok=True)

        # set model
        models = {
            "UNET": lambda: UNet(n_channels=3, n_classes=self.n_classes, bilinear=self.bilinear),
            "R2UNET": lambda: R2UNet(n_channels=3, n_classes=self.n_classes, bilinear=self.bilinear),
            "MASKRCNN": lambda: MaskRCNNResNet50(n_classes=self.n_classes)
        }
        self.model = models[self.model_name]() if self.model_name in models else None
        assert self.model is not None, f"Model {self.model_name} not recognized"
        self.model = self.model.to(device=self.device, memory_format=torch.channels_last)

        # data information
        self.n_train = 0
        self.n_val = 0

        logging.info(f'Network: \n'
                     f'\t{self.model.in_features} input channels\n'
                     f'\t{self.model.n_classes} output channels (n_classes)\n')

    def train_and_validate(self):
        # data loader
        train_loader, val_loader = self.set_train_val_dataloader()

        # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # optimizer
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # change lr by indicators
        scheduler = CyclicLR(optimizer, base_lr=1e-8, max_lr=1e-3, step_size_up=50, mode='triangular', cycle_momentum=False)  # change lr by epoch
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        early_stopping = EarlyStopping(patience=5, verbose=True)  # set early stop

        # Start Train
        self.train_loop(train_loader, val_loader, optimizer, scheduler, grad_scaler, early_stopping)

    def train_loop(self, train_loader, val_loader, optimizer, scheduler, grad_scaler, early_stopping):
        global_step = 0  # Record global step
        best_dice = 0  # Record the best dice

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            with (tqdm(total=self.n_train, desc=f'Epoch {epoch}/{self.epochs}', unit='img') as pbar):
                for batch in train_loader:
                    # Set image and targets
                    images, target = self.reset_image_target(batch['image'], batch['target'])
                    with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
                        loss_dict = self.model(images, target)
                        losses = sum(loss for loss in loss_dict.values()) + loss_dict['loss_mask'] * 5  # if maskrcnn

                    optimizer.zero_grad()
                    grad_scaler.scale(losses).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clipping)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    scheduler.step()

                    pbar.update(images.shape[0])
                    current_lr = optimizer.param_groups[0]['lr']
                    pbar.set_postfix(**{**{k: v.item() for k, v in loss_dict.items()}, 'lr': current_lr})
                    global_step += 1

                val_indicators = self.evaluate_loop(val_loader)
                dice_score = val_indicators['segmentation_metrics']['dice']

                if self.segmentation_task:
                    logging.info('Segmentation Metrics:')
                    for key, value in val_indicators['segmentation_metrics'].items():
                        logging.info(f'  {key}: {value}')
                if self.detection_task:
                    logging.info('Detection Metrics:')
                    for key, value in val_indicators['detection_metrics'].items():
                        logging.info(f'  {key}: {value}')

                # update early stopping
                stop = early_stopping(dice_score)
                if stop:
                    break

                indicators_save_path = f'{self.indicators_save_dir}/indicators.csv'
                with open(indicators_save_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    # If the file does not exist or is empty, write the column header
                    if not os.path.isfile(indicators_save_path):
                        writer.writerow(val_indicators.keys())
                    writer.writerow(val_indicators.values())

                # save model
                state_dict = self.model.state_dict()
                if dice_score > best_dice:
                    best_dice = dice_score
                    torch.save(state_dict, str(self.checkpoint_save_dir / f'best{best_dice}.pth'))
                    logging.info(f'Checkpoint best dice({best_dice}) saved!')
                torch.save(state_dict, str(self.checkpoint_save_dir / f'last.pth'))

    def evaluate_loop(self, val_loader):
        self.model.eval()
        # Initialize cumulative indicator
        metrics = {}
        sum_seg_metrics = None
        sum_detection_metrics = None
        total_samples = 0

        # Data required to initialize the segmentation task

        # iterate over the validation set
        with torch.no_grad():
            with ((torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp))):
                for batch in tqdm(val_loader, total=len(val_loader), desc='Validation round', unit='batch', leave=False):
                    images, target_dict = batch['image'], batch['target']
                    images = images.to(self.device)
                    outputs = self.model(images)

                    # Collect forecasts and real data
                    for i, output in enumerate(outputs):
                        if self.segmentation_task:
                            # Segmentation task indicator calculation
                            segmentation_pred = output['masks']
                            merged_pred_mask = merge_masks(segmentation_pred)

                            segmentation_true = target_dict['masks'][i]

                            batch_seg_metrics = segmentation_indicators(
                                input=merged_pred_mask.to('cpu'),
                                target=segmentation_true.to('cpu'),
                                multi_class=True
                            )

                            # cumulative indicator
                            if sum_seg_metrics is None:
                                sum_seg_metrics = batch_seg_metrics
                            else:
                                sum_seg_metrics = {k: sum_seg_metrics[k] + batch_seg_metrics[k] for k in sum_seg_metrics}
                        if self.detection_task:
                            # Indicator calculation for detection tasks
                            pred_boxes = output['boxes'].to('cpu')
                            pred_scores = output['scores'].to('cpu')
                            pred_labels = output['labels'].to('cpu')
                            true_boxes = target_dict['boxes'][i].to('cpu')
                            true_labels = target_dict['labels'][i].to('cpu')
                            batch_detect_metrics = detection_indicators(
                                pred_boxes=pred_boxes,
                                true_boxes=true_boxes,
                                pred_scores=pred_scores,
                                pred_labels=pred_labels,
                                true_labels=true_labels

                            )

                            if sum_detection_metrics is None:
                                sum_detection_metrics = batch_detect_metrics
                            else:
                                sum_detection_metrics = {k: sum_detection_metrics[k] + batch_detect_metrics[k] for k in
                                                         sum_detection_metrics}

                        total_samples += 1
        if self.segmentation_task:
            avg_seg_metrics = {k: v / total_samples for k, v in sum_seg_metrics.items()}
            metrics['segmentation_metrics'] = avg_seg_metrics
        if self.detection_task:
            avg_detection_metrics = {k: v / total_samples for k, v in sum_detection_metrics.items()}
            metrics['detection_metrics'] = avg_detection_metrics

        self.model.train()
        return metrics

    def set_train_val_dataloader(self):
        # Create dataset
        dataset = SegmentDatasetJson(self.image_dir, self.json_dir, size=self.size)

        # Split into train / validation partitions
        val_percent = self.validation / 100
        self.n_val = int(len(dataset) * val_percent)
        self.n_train = len(dataset) - self.n_val
        train_set, val_set = random_split(dataset, [self.n_train, self.n_val], generator=torch.Generator().manual_seed(0))

        # Create dataloader os.cpu_count()
        loader_args = dict(batch_size=self.batch_size, num_workers=4, pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

        logging.info(f'''Starting training:
                    Epochs:          {self.epochs}
                    Batch size:      {self.batch_size}
                    Learning rate:   {self.learning_rate}
                    Training size:   {self.n_train}
                    Validation size: {self.n_val}
                    Device:          {self.device.type}
                    Images size:  {self.size}
                    Mixed Precision: {self.amp}
                ''')

        return train_loader, val_loader

    def reset_image_target(self, images, target_dict):
        images = images.to(self.device)
        targets = []
        for i in range(len(images)):
            single_target = {k: v[i].to(self.device) for k, v in target_dict.items()}
            targets.append(single_target)

        return images, targets


if __name__ == "__main__":
    segment_train = Segmentation_Trainer()
    segment_train.train_and_validate()
