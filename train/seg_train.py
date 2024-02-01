import argparse
import csv
import logging

from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from seg_evaluate import evaluate
from models.seg.UNet.unet_model import UNet, R2UNet
from utils.data_loading import SegmentDatasetJson, SegmentationDatasetDirectory
from models.seg.MaskRCNN.model import MaskRCNNResNet50
from utils.indicators import segmentation_indicators
from utils.utils import EarlyStopping

image_dir = 'E:/Datas/work/HairEffect/SegmentData/DEMO_IMAGES'
mask_dir = 'E:/Datas/work/HairEffect/SegmentData/DEMO_MASKS'


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images_0 and target masks_f')
    parser.add_argument('--model_name', '-m', type=str, default='UNET', help="select train model_name")
    parser.add_argument('--cycle-times', '-t', type=int, default=4, help='cycle times in R2UNet')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--size', '-s', type=float, default=(256, 256), help='Downscaling factor of the images_0')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upSampling')
    parser.add_argument('--n_classes', '-c', type=int, default=2, help='Number of n_classes')

    return parser.parse_args()


def train_model(
        model,
        model_name,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        size: tuple = (128, 128),
        amp: bool = False,
        gradient_clipping: float = 1.0,
):

    # set save directory
    now_time = datetime.now().strftime("%Y-%m-%d")  # Get the current time in a specific format
    now_h = datetime.now().strftime("%H")

    dir_checkpoint = Path(f'./checkpoints/{model_name}')
    dir_indicators = Path(f'./indicators/{model_name}')
    Path(f'{dir_checkpoint}/{now_time}/{now_h}').mkdir(parents=True, exist_ok=True)
    Path(f'{dir_indicators}/{now_time}/{now_h}').mkdir(parents=True, exist_ok=True)
    dir_checkpoint_save = Path(f'{dir_checkpoint}/{now_time}/{now_h}')
    dir_indicators_save = Path(f'{dir_indicators}/{now_time}/{now_h}')

    # Create dataset
    dataset = SegmentationDatasetDirectory(image_dir, mask_dir, size, [0, 255])

    # Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # Create data loaders os.cpu_count()
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # Initialize logging
    experiment = wandb.init(project=model_name, resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, size=size, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images size:  {size}
        Mixed Precision: {amp}
    ''')

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    early_stopping = EarlyStopping(patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    best_dice = 0

    # save indicators in csv
    indicators_header = ['epoch', 'train_loss', 'train_dice', 'train_iou', 'train_f1', 'train_recall',
                         'train_precision',
                         'val_dice', 'val_iou', 'val_f1', 'val_recall', 'val_precision', 'learning_rate']
    epoch_csv_path = f"{dir_indicators_save}/train_indicators.csv"

    with open(epoch_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(indicators_header)

    # Begin training
    for epoch in range(1, epochs + 1):
        model.train()

        # save epoch train indicators
        epoch_train_loss = np.array([])
        epoch_train_dice = np.array([])
        epoch_train_iou = np.array([])
        epoch_train_f1 = np.array([])
        epoch_train_recall = np.array([])
        epoch_train_precision = np.array([])
        # save epoch val
        epoch_val_dice = np.array([])
        epoch_val_iou = np.array([])
        epoch_val_f1 = np.array([])
        epoch_val_recall = np.array([])
        epoch_val_precision = np.array([])

        # Record the number of verifications performed in each epoch
        val_times_every_epoch = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['images'], batch['masks']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images_0 have {images.shape[1]} channels. Please check that ' \
                    'the images_0 are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        train_indicators_dict = segmentation_indicators(F.sigmoid(masks_pred.squeeze(1)),
                                                                        true_masks.float(), multi_class=False,
                                                                        reduce_batch_first=True, places=3)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        train_indicators_dict = segmentation_indicators(F.softmax(masks_pred, dim=1).float(),
                                                                        F.one_hot(true_masks, model.n_classes).
                                                                        permute(0, 3, 1, 2).float(), multi_class=True,
                                                                        reduce_batch_first=True, places=3)
                d_loss = 1 - train_indicators_dict['dice']
                loss += d_loss
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                pbar.update(images.shape[0])

                # update epoch information
                global_step += 1
                epoch_train_loss = np.append(epoch_train_loss, loss.cpu().detach().numpy())
                epoch_train_dice = np.append(epoch_train_dice, train_indicators_dict['dice'])
                epoch_train_iou = np.append(epoch_train_iou, train_indicators_dict['iou'])
                epoch_train_f1 = np.append(epoch_train_f1, train_indicators_dict['f1'])
                epoch_train_recall = np.append(epoch_train_recall, train_indicators_dict['recall'])
                epoch_train_precision = np.append(epoch_train_precision, train_indicators_dict['precision'])
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round every 5 batch size do a evaluation
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        val_times_every_epoch += 1

                        val_score = evaluate(model, val_loader, device, amp)
                        # score every batch
                        val_dice, val_iou, val_f1 = val_score[0], val_score[1], val_score[2]
                        val_recall, val_precision = val_score[3], val_score[4]

                        #  average score for every valuate
                        val_dice_avg = sum(val_dice) / max(len(val_loader), 1)
                        val_iou_avg = sum(val_iou) / max(len(val_loader), 1)
                        val_f1_avg = sum(val_f1) / max(len(val_loader), 1)
                        val_recall_avg = sum(val_recall) / max(len(val_loader), 1)
                        val_precision_avg = sum(val_precision) / max(len(val_loader), 1)

                        # score for every batch in every epoch
                        epoch_val_dice = np.append(epoch_val_dice, val_dice_avg)
                        epoch_val_iou = np.append(epoch_val_iou, val_iou_avg)
                        epoch_val_f1 = np.append(epoch_val_f1, val_f1_avg)
                        epoch_val_recall = np.append(epoch_val_recall, val_recall_avg)
                        epoch_val_precision = np.append(epoch_val_precision, val_precision_avg)

                        logging.info('Validation Dice score: {}'.format(np.round(val_dice_avg, 3)))
                        logging.info('Validation Iou score: {}'.format(np.round(val_iou_avg, 3)))
                        logging.info('Validation f1 score: {}'.format(np.round(val_f1_avg, 3)))
                        logging.info('Validation Recall score: {}'.format(np.round(val_recall_avg, 3)))
                        logging.info('Validation Precision score: {}'.format(np.round(val_precision_avg, 3)))

                        scheduler.step(val_dice_avg)
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_dice_avg,
                            'validation Iou': val_iou_avg,
                            'validation F1': val_f1_avg,
                            'validation Recall': val_recall_avg,
                            'validation Precision': val_precision_avg,
                            'images_0': wandb.Image(images[0].cpu()),
                            'masks_f': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch
                        })

            # Validation is performed every 1/5 of training
            stage = len(train_loader)
            print(stage)
            train_loss_avg = np.round((sum(epoch_train_loss) / stage), 3)
            train_dice_avg = np.round((sum(epoch_train_dice) / stage), 3)
            train_iou_avg = np.round((sum(epoch_train_iou) / stage), 3)
            train_f1_avg = np.round((sum(epoch_train_f1) / stage), 3)
            train_recall_avg = np.round((sum(epoch_train_recall) / stage), 3)
            train_precision_avg = np.round((sum(epoch_train_precision) / stage), 3)

            epoch_val_dice_avg = np.round((sum(epoch_val_dice) / val_times_every_epoch), 3)
            epoch_val_iou_avg = np.round((sum(epoch_val_iou) / val_times_every_epoch), 3)
            epoch_val_f1_avg = np.round((sum(epoch_val_f1) / val_times_every_epoch), 3)
            epoch_val_recall_avg = np.round((sum(epoch_val_recall) / val_times_every_epoch), 3)
            epoch_val_precision_avg = np.round((sum(epoch_val_precision) / val_times_every_epoch), 3)

            stop = early_stopping(epoch_val_dice_avg)
            if stop:
                break

        epoch_row = [epoch, train_loss_avg, train_dice_avg, train_iou_avg, train_f1_avg,
                     train_recall_avg, train_precision_avg, epoch_val_dice_avg, epoch_val_iou_avg, epoch_val_f1_avg,
                     epoch_val_recall_avg, epoch_val_precision_avg, optimizer.param_groups[0]['lr']]

        with open(epoch_csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(epoch_row)

        if save_checkpoint:
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            if epoch_val_dice_avg > best_dice:
                best_dice = epoch_val_dice_avg
                torch.save(state_dict, str(dir_checkpoint_save / f'best.pth'))
                logging.info(f'Checkpoint best dice({best_dice}) saved!')
            torch.save(state_dict, str(dir_checkpoint_save / f'last.pth'))

    logging.info(f'save time: {now_time}')


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    logging.info(f'Using model_name {args.model_name}')

    # n_channels=3 for RGB images_0
    # n_classes is the number of probabilities you want to get per pixel

    all_models = ['UNET', 'R2UNET', 'MASKRCNN']
    model = None
    assert args.model_name.upper() in all_models, f"model_name optional type are: {all_models}"

    if args.model_name.upper() == "UNET":
        model = UNet(n_channels=3, n_classes=args.n_classes, bilinear=args.bilinear)
    elif args.model_name.upper() == "R2UNET":
        assert args.cycle_times is not None, "when choose R2UNET must set cycle-times"
        model = R2UNet(n_channels=3, n_classes=args.n_classes, t=args.cycle_times, bilinear=args.bilinear)
    elif args.model_name.upper() == "MASKRCNN":
        model = MaskRCNNResNet50(n_classes=args.n_classes)

    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network: \n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (n_classes)\n')

    model.to(device=device)
    try:
        train_model(
            model=model,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            size=args.size,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
