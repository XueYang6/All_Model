import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.indicators import segmentation_indicators


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    val_dice = []
    val_iou = []
    val_f1 = []
    val_recall = []
    val_precision = []

    # iterate over the validation set
    with ((torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp))):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['images'], batch['masks']

            # move images_0 and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                indicators_dict = segmentation_indicators(mask_pred.squeeze(1), mask_true, multi_class=False,
                                                          reduce_batch_first=False, places=3)
                val_dice.append(indicators_dict['dice'])
                val_iou.append(indicators_dict['iou'])
                val_f1.append(indicators_dict['f1'])
                val_recall.append(indicators_dict['recall'])
                val_precision.append(indicators_dict['precision'])

                conf_matrix = 0

            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, \
                    'True mask indices should be in [0, n_classes]'
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the indicators, ignoring background
                indicators_dict = segmentation_indicators(mask_pred, mask_true, multi_class=True,
                                                          reduce_batch_first=False, places=3)
                val_dice.append(indicators_dict['dice'])
                val_iou.append(indicators_dict['iou'])
                val_f1.append(indicators_dict['f1'])
                val_recall.append(indicators_dict['recall'])
                val_precision.append(indicators_dict['precision'])

    net.train()

    return [val_dice, val_iou, val_f1, val_recall, val_precision, conf_matrix]
