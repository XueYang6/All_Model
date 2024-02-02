import torch
import numpy as np

from sklearn.metrics import precision_recall_curve, auc, f1_score

from torch import Tensor


def segmentation_indicators(input: Tensor, target: Tensor, multi_class: bool = False, reduce_batch_first: bool = False,
                            epsilon: float = 1e-6, places: int = 4):
    """
    Calculate segmentation metrics including dice, iou, f1, precision, and recall.

    Args:
        input (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Ground truth tensor.
        multi_class (bool, optional): Whether to handle multi-class segmentation. Defaults to False.
        reduce_batch_first (bool, optional): Whether to reduce along the batch dimension first. Defaults to False.
        epsilon (float, optional): Small value to avoid division by zero. Defaults to 1e-6.
        places (int, optional): Decimal places for rounding the metrics. Defaults to 4.

    Returns:
        dict: Dictionary containing calculated metrics.
    """

    # Validate input dimensions
    assert input.size() == target.size(), f'input size{input.size()} must same as {target.size()}'

    if reduce_batch_first:
        if multi_class:
            num_classes = input.shape[1]
            batch_size, _, height, width = input.shape
            input = input.permute(1, 0, 2, 3).reshape(num_classes, -1, height, width)
            target = target.permute(1, 0, 2, 3).reshape(num_classes, -1, height, width)
        else:
            # For binary segmentation, flatten the batch and height*width dimensions
            input = input.reshape(-1, input.size(-2) * input.size(-1))
            target = target.reshape(-1, target.size(-2) * target.size(-1))

    metrics = {'dice': [], 'iou': [], 'f1': [], 'precision': [], 'recall': []}
    for cls in range(input.shape[1] if multi_class else 1):
        if multi_class:
            input_cls = input[:, cls, :, :]
            target_cls = target[:, cls, :, :]
        else:
            input_cls = input
            target_cls = target

        tp = (input_cls * target_cls).sum(dim=(-2, -1))
        fp = input_cls.sum(dim=(-2, -1)) - tp
        fn = target_cls.sum(dim=(-2, -1)) - tp

        precision = (tp + epsilon) / (tp + fp + epsilon)
        recall = (tp + epsilon) / (tp + fn + epsilon)
        dice = (2 * tp + epsilon) / (tp + tp + fp + fn + epsilon)
        iou = (tp + epsilon) / (tp + fp + fn + epsilon)
        f1_score = (2 * precision * recall) / (precision + recall + epsilon)

        metrics['dice'].append(dice.mean().item())
        metrics['iou'].append(iou.mean().item())
        metrics['f1'].append(f1_score.mean().item())
        metrics['precision'].append(precision.mean().item())
        metrics['recall'].append(recall.mean().item())

        # Average the metrics across classes
    averaged_metrics = {k: np.round(np.mean(v), places) for k, v in metrics.items()}

    return averaged_metrics


def box_iou(box1, box2):
    """
    Calculate the IoU of two bounding boxes

    Args:
        box1 (list or torch.Tensor): first bounding box [x_min, y_min, x_max, y_max].
        box2 (list or torch.Tensor): second bounding box [x_min, y_min, x_max, y_max].
    Returns:
        float: IoU value
    """
    # Verify bounding box format
    assert (box1[0] < box1[2] and box1[1] < box1[3]), f"box1: {box1} not in format [x_min, y_min, x_max, y_max]"
    assert (box2[0] < box2[2] and box2[1] < box2[3]), f"box2: {box2} not in format [x_min, y_min, x_max, y_max]"

    # Calculate the coordinates of the intersection rectangle
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    # Calculate the area of intersection
    inter_area = max(0, x_max - x_min) * max(0, y_max - y_min)

    # Calculate the area of the respective rectangle
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union area and IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def detection_indicators(pred_boxes, true_boxes, pred_scores, true_labels, pred_labels, iou_threshold=0.5, places=4):
    """
    Calculate detection metrics including AP, precision, recall and F1.

    Args:
        pred_boxes (torch.Tensor): Predicted bounding boxes.
        true_boxes (torch.Tensor): Ground truth bounding boxes.
        pred_scores (torch.Tensor): Predicted bounding box scores.
        true_labels (torch.Tensor): Ground truth labels.
        pred_labels (torch.Tensor): Predicted labels.
        iou_threshold (float): IoU threshold for determining true positives.
        places (int): Decimal places for rounding the metrics.

    Returns:
        dict: Dictionary containing calculated metrics.
    """

    # initial indicators
    all_ap, all_precision, all_recall, all_f1 = [], [], [], []

    # Calculate metrics for each category
    for class_id in true_labels.unique():
        # Extract predicted and real data for specific categories
        class_pred_inds = pred_labels == class_id
        class_pred_boxes = pred_boxes[class_pred_inds]
        class_pred_socre = pred_scores[class_pred_inds]
        class_true_boxes = true_boxes[true_labels == class_id]

        if class_true_boxes.numel() == 0:
            continue  # If there is no real box for this category, skip

        # initial labels and scores list
        labels, scores = [], []

        # For each ground truth box, find the best matching predicted box
        for true_box in class_true_boxes:
            max_iou, max_score = 0, 0
            for pred_box, score in zip(class_pred_boxes, class_pred_socre):
                iou = box_iou(pred_box, true_box)
                if iou > max_iou:
                    max_iou, max_score = iou, score

            if max_iou > iou_threshold:
                labels.append(1)
            else:
                labels.append(0)
            scores.append(max_score)

        # Calculate the average accuracy for each category
        labels = np.array(labels)
        scores = np.array(scores)

        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = auc(recall, precision)
        all_ap.append(ap)

        # calculate other indicators
        if len(precision) > 0 and len(recall) > 0:
            all_precision.append(precision[-1])
            all_recall.append(recall[-1])
            f1 = f1_score(labels, scores >= 0)
            all_f1.append(f1)

    # Calculate the average of all categories
    m_ap = np.round(np.mean(all_ap) if all_ap else 0.0, places)
    m_precision = np.round(np.mean(all_precision) if all_precision else 0.0, places)
    m_recall = np.round(np.mean(all_recall) if all_recall else 0.0, places)
    m_f1 = np.round(np.mean(all_f1) if all_f1 else 0.0, places)

    return {
        'mAP': m_ap,
        'precision': m_precision,
        'recall': m_recall,
        'F1': m_f1
    }



