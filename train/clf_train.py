import csv
import logging
import argparse
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import wandb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from clf_evaluate import evaluate_model
from utils.data_loading import ClassificationDatasetJson
from utils.utils import EarlyStopping, stratified_split, ToHSV, ApplyCLAHE
from models.clf.ResNet.resnet import ResNetClassification
from models.clf.EfficientNet.model import EfficientNetClassification


MODEL = ['RES-NET', 'EFFICIENT-NET']

json_data_path = '../datasets/json/MelNvTrain_dataset.json'


def get_args():
    parser = argparse.ArgumentParser(description='Train the  model_name')
    parser.add_argument('--model_name', '-m', type=str, default='EFFICIENT-NET',
                        help="select train model_name")
    parser.add_argument('--size', '-s', type=tuple, default=(256, 256),
                        help='set images size')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--val', '-vp', metavar='VP', type=float, default=20,
                        help='Percentage of data for validation')
    parser.add_argument('--num-classes', '-n', type=int, default=2,
                        help='num of n_classes')
    return parser.parse_args()


# Function to train the classification model_name
def classification_train(
        model,
        num_classes,
        model_name,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.2,
        save_checkpoint: bool = True,
        size: tuple = (256, 256),
        amp: bool = False,
        gradient_clipping: float = 1.0,
):
    # set save directory
    now_time = datetime.now().strftime("%Y-%m-%d")  # Get the current time in a specific format
    now_h = datetime.now().strftime("%H")

    dir_checkpoint = Path(f'./train_save/MelNv/checkpoints/{model_name}_b1')
    dir_indicators = Path(f'./train_save/MelNv/indicators/{model_name}_b1')
    Path(f'{dir_checkpoint}/{now_time}/{now_h}').mkdir(parents=True, exist_ok=True)
    Path(f'{dir_indicators}/{now_time}/{now_h}').mkdir(parents=True, exist_ok=True)
    dir_checkpoint_save = Path(f'{dir_checkpoint}/{now_time}/{now_h}')
    dir_indicators_save = Path(f'{dir_indicators}/{now_time}/{now_h}')

    # transform
    transform = transforms.Compose([
        transforms.Resize(size),
        ApplyCLAHE(),
        transforms.ToTensor(),
    ])

    # Create dataset
    dataset = ClassificationDatasetJson(json_dir=json_data_path, size=size, transform=transform)
    classes = dataset.get_classes()
    assert num_classes == len(classes), f'The output category of the model_name does not match the category in the json data'

    # split into train/validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # Initialize logging
    experiment = wandb.init(project=model_name, resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, size=size, amp=amp)
    )

    logging.info(f'''Start training:
    Epochs:             {epochs}
    Batch size:         {batch_size}
    Learning rate:      {learning_rate}
    Training size:      {n_train}
    Validation size:    {n_val}
    Checkpoints:        {save_checkpoint}
    Device:             {device.type}
    Images size:        {size}
    Mixed Precision:    {amp}
    ''')

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    best_acc = 0.0

    # set and save header indicators in csv
    indicators_header = ['epoch', 'train_loss', 'learning_rate',
                         'train_acc', 'train_precision', 'train_recall', 'train_f1', 'train_mAP',
                         'val_acc', 'val_precision', 'val_recall', 'val_f1', 'val_mAP']
    epoch_csv_path = f'{dir_indicators_save}/train_indicators.csv'
    with open(epoch_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(indicators_header)

    # Begin training
    for epoch in range(1, epochs + 1):
        model.train()

        val_times_every_epoch = 0  # Record the number of verifications performed in each epoch

        # save epoch indicators
        epoch_train_loss = 0
        epoch_train_acc = 0
        epoch_train_precision = 0
        epoch_train_recall = 0
        epoch_train_f1 = 0
        epoch_train_ap = 0

        epoch_val_acc = 0
        epoch_val_precision = 0
        epoch_val_recall = 0
        epoch_val_f1 = 0
        epoch_val_ap = 0

        with (tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='image') as pbar):
            for batch in train_loader:
                images, true_labels = batch['images'], batch['labels']

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_labels = true_labels.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else ' cpu', enabled=amp):
                    proba_labels = model(images)
                    pred_labels = torch.argmax(F.softmax(proba_labels, dim=1), dim=1)

                    loss = criterion(proba_labels, true_labels)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                pbar.update(images.shape[0])

                # calculate indicators
                true_labels_cpu = true_labels.cpu().numpy()
                proba_labels_cpu = proba_labels.detach().cpu().numpy()
                pred_labels_cpu = pred_labels.detach().cpu().numpy()

                if num_classes > 2:
                    train_acc = accuracy_score(true_labels_cpu, pred_labels_cpu, normalize=True)
                    train_precision = precision_score(true_labels_cpu, pred_labels_cpu, average='macro', zero_division=0)
                    train_recall = recall_score(true_labels_cpu, pred_labels_cpu, average='macro', zero_division=0)
                    train_f1 = f1_score(true_labels_cpu, pred_labels_cpu, average='macro', zero_division=0)
                    train_ap = 0
                else:
                    train_acc = accuracy_score(true_labels_cpu, pred_labels_cpu, normalize=True)
                    train_precision = precision_score(true_labels_cpu, pred_labels_cpu, average='binary', zero_division=0)
                    train_recall = recall_score(true_labels_cpu, pred_labels_cpu, zero_division=0)
                    train_f1 = f1_score(true_labels_cpu, pred_labels_cpu, zero_division=0)
                    train_ap = average_precision_score(true_labels_cpu, proba_labels_cpu[:, 1])

                # update epoch information
                global_step += 1
                epoch_train_loss += loss.cpu().detach().numpy()
                epoch_train_acc += train_acc
                epoch_train_precision += train_precision
                epoch_train_recall += train_recall
                epoch_train_f1 += train_f1
                epoch_train_ap += train_ap
                experiment.log({
                    'epoch': epoch,
                    'step': global_step,
                    'train loss': loss.item(),
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Start validation evaluate
                division_step = (n_train // (5 * batch_size))  # Evaluation round every 5 batch size do a evaluation
                if (global_step % division_step == 0) & (division_step > 0):
                    val_times_every_epoch += 1

                    # indicators in every evaluate
                    val_acc, val_precision, val_recall, val_f1, val_ap = evaluate_model(model, val_loader, device, amp, num_classes)

                    # average score for every evaluate
                    val_acc_avg = sum(val_acc) / max(len(val_loader), 1)
                    val_precision_avg = sum(val_precision) / max(len(val_loader), 1)
                    val_recall_avg = sum(val_recall) / max(len(val_loader), 1)
                    val_f1_avg = sum(val_f1) / max(len(val_loader), 1)
                    val_ap_avg = sum(val_ap) / max(len(val_loader), 1)

                    logging.info('Validation ACC score: {}'.format(np.round(val_acc_avg, 3)))
                    logging.info('Validation Precision score: {}'.format(np.round(val_precision_avg, 3)))
                    logging.info('Validation Recall score: {}'.format(np.round(val_recall_avg, 3)))
                    logging.info('Validation F1 score: {}'.format(np.round(val_f1_avg, 3)))
                    logging.info('Validation AP score: {}'.format(np.round(val_ap_avg, 3)))

                    # score add in epoch indicators
                    epoch_val_acc += val_acc_avg
                    epoch_val_precision += val_precision_avg
                    epoch_val_recall += val_recall_avg
                    epoch_val_f1 += val_f1_avg
                    epoch_val_ap += val_ap_avg

                    # update scheduler
                    scheduler.step(val_recall_avg)
                    experiment.log({
                        'Learning Rate': optimizer.param_groups[0]['lr'],
                        'Validation ACC': val_acc_avg,
                        'Validation Precision': val_precision_avg,
                        'Validation Recall': val_recall_avg,
                        'Validation F1': val_f1_avg,
                        'Validation AP': val_ap_avg,
                        'images_0': wandb.Image(images[0].cpu()),
                        'label': pred_labels[0].int().cpu(),
                        'step': global_step,
                        'epoch': epoch
                    })

        # calculate indicators average score in every epoch
        epoch_train_loss_avg = np.round((epoch_train_loss / len(train_loader)), 3)
        epoch_train_acc_avg = np.round((epoch_train_acc / len(train_loader)), 3)
        epoch_train_precision_avg = np.round((epoch_train_precision / len(train_loader)), 3)
        epoch_train_recall_avg = np.round((epoch_train_recall / len(train_loader)), 3)
        epoch_train_f1_avg = np.round((epoch_train_f1 / len(train_loader)), 3)
        epoch_train_ap_avg = np.round((epoch_train_ap / len(train_loader)), 3)

        epoch_val_acc_avg = np.round((epoch_val_acc / val_times_every_epoch), 3)
        epoch_val_precision_avg = np.round((epoch_val_precision / val_times_every_epoch), 3)
        epoch_val_recall_avg = np.round((epoch_val_recall / val_times_every_epoch), 3)
        epoch_val_f1_avg = np.round((epoch_val_f1 / val_times_every_epoch), 3)
        epoch_val_ap_avg = np.round((epoch_val_ap / val_times_every_epoch), 3)

        # set early stop
        stop = early_stopping(epoch_val_acc_avg)
        if stop:
            break

        # write indicators in csv
        epoch_row = [epoch, epoch_train_loss_avg, optimizer.param_groups[0]['lr'], epoch_train_acc_avg,
                     epoch_train_precision_avg, epoch_train_recall_avg, epoch_train_f1_avg, epoch_train_ap_avg,
                     epoch_val_acc_avg, epoch_val_precision_avg, epoch_val_recall_avg,
                     epoch_val_f1_avg, epoch_val_ap_avg]

        with open(epoch_csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(epoch_row)

        if save_checkpoint:
            state_dict = model.state_dict()
            state_dict['n_classes'] = classes
            if epoch_val_acc_avg > best_acc:
                best_acc = epoch_val_acc_avg
                torch.save(state_dict, str(dir_checkpoint_save / 'best.pth'))
                logging.info(f'Checkpoint best acc{best_acc} saved!')
            torch.save(state_dict, str(dir_checkpoint_save / f'last.pth'))

        logging.info('Finished hole train')


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    logging.info(f'Using model_name {args.model_name}')

    model = None
    assert args.model_name.upper() in MODEL, f'Model optional type are :{MODEL}, please choose available model_name'

    if args.model_name.upper() == 'RES-NET':
        model = ResNetClassification(args.num_classes)
    elif args.model_name.upper() == 'EFFICIENT-NET':
        model = EfficientNetClassification(in_channels=3, num_classes=args.num_classes)

    model = model.to(device=device)

    logging.info(f'Network:\n'
                 f'\t{args.num_classes} output channels (n_classes)\n')

    try:
        classification_train(
            model=model,
            model_name=args.model_name,
            num_classes=args.num_classes,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_percent=args.val / 100,
            save_checkpoint=True,
            size=args.size,
            amp=args.amp,
            gradient_clipping=1.0,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')



