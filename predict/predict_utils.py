from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data_loading import preprocess


def predict_mask(model: nn.Module, image: Image, device, size=(256, 256), num_classes=1, threshold=0.5,
                 mask_values=None):
    mask_values = [0, 255] if mask_values is None else mask_values
    model.eval()
    image = torch.from_numpy(preprocess(image, size, is_mask=False, mask_values=mask_values))
    image = image.unsqueeze(0)
    image = image.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = model(image).cpu()

        output = F.interpolate(output, size, mode='bilinear')

        if num_classes > 1:
            mask = output.argmax(dim=1)
            proba = output
        else:
            mask = torch.sigmoid(output) > threshold
            proba = torch.sigmoid(output)

    return mask[0].long().squeeze().numpy(), proba[0].float().squeeze().numpy()


def predict_to_mask(predict: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((predict.shape[-2], predict.shape[-1], len(mask_values[0])), dtype=np.uint8)
    else:
        out = np.zeros((predict.shape[-2], predict.shape[-1]), dtype=np.uint8)
    if predict.ndim == 3:
        predict = np.argmax(predict, axis=0)

    for i, v in enumerate(mask_values):
        out[predict == i] = v

    return Image.fromarray(out)
