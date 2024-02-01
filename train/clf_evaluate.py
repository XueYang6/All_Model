import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_model(model, dataloader, device, amp, num_classes):
    model.eval()
    num_batches = len(dataloader)
    val_acc = []
    val_precision = []
    val_recall = []
    val_f1 = []
    val_ap = []

    with torch.no_grad():
        for batch in tqdm(dataloader, total=num_batches, desc='Validation round', unit='batch', leave=False):
            images, labels = batch['images'], batch['labels']
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_labels = labels.to(device=device, dtype=torch.long)

            with ((torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp))):
                proba_labels = model(images)
                pred_labels = torch.argmax(F.softmax(proba_labels, dim=1), dim=1)

            # calculate the val indicators
            true_labels_cpu = true_labels.cpu().numpy()
            proba_labels_cpu = proba_labels.detach().cpu().numpy()
            pred_labels_cpu = pred_labels.detach().cpu().numpy()

            if num_classes > 2:
                val_acc.append(accuracy_score(true_labels_cpu, pred_labels_cpu, normalize=True))
                val_precision.append(precision_score(true_labels_cpu, pred_labels_cpu, average='macro', zero_division=0))
                val_recall.append(recall_score(true_labels_cpu, pred_labels_cpu, average='macro', zero_division=0))
                val_f1.append(f1_score(true_labels_cpu, pred_labels_cpu, average='macro'))
                val_ap.append(0)
            else:
                val_acc.append(accuracy_score(true_labels_cpu, pred_labels_cpu))
                val_precision.append(precision_score(true_labels_cpu, pred_labels_cpu))
                val_recall.append(recall_score(true_labels_cpu, pred_labels_cpu))
                val_f1.append(f1_score(true_labels_cpu, pred_labels_cpu))
                val_ap.append(average_precision_score(true_labels_cpu, proba_labels_cpu[:, 1]))

    return val_acc, val_precision, val_recall, val_f1, val_ap


