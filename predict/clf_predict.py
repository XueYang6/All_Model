import os
import logging
from datetime import datetime

import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             average_precision_score, confusion_matrix)

from models.clf.EfficientNet.model import EfficientNetClassification
from models.clf.ResNet.resnet import ResNetClassification
from utils.data_loading import ClassificationDatasetJson
from utils.utils import std_mpl, draw_roc, draw_confusion_matrix, draw_grad_cam

now_time = datetime.now().strftime("%Y-%m-%d")
now_h = datetime.now().strftime("%H")

MODEL = ['RES-NET', 'EFFICIENT-NET']

model_path = '../train/MelNv/checkpoints/RES-NET/2024-01-27/22/best.pth'
test_json_path = '../datas/json_data/MelNvWithTrain_dataset.json'
output_path = 'Output/MelNv/WithHair'


def get_args():
    parser = argparse.ArgumentParser(description='Predict labels from input images')
    parser.add_argument('--model_name', '-m', type=str, nargs='+',
                        default='RES-NET', help='Specify the file in which the model is stored')
    parser.add_argument('--load-model', '-lm', metavar='FILE', nargs='+',
                        default=model_path, help='Specify the file in which the model is stored')
    parser.add_argument('--size', '-s', type=tuple,
                        default=(256, 256), help='Scale factor for the input images')
    parser.add_argument('--n_classes', '-c', type=int,
                        default=2, help='Number of n_classes')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        default=test_json_path, help='Json filename of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+',
                        default=output_path, help='File name of output images')
    parser.add_argument('--viz', '-v',
                        default=True, help='Whether to visualize test indicators')
    parser.add_argument('--cam-value', '-gv', type=int,
                        default=0, help='saliency map values (2 means 2 and 1)')
    parser.add_argument('--threshold', '-t', type=float,
                        default=0.5, help='Minimum probability value ro consider a mask pixel white')
    parser.add_argument('--amp', '-a', type=bool,
                        default=True)

    args = parser.parse_args()
    return args


def tensor2pil(tensor):
    """
    Convert a PyTorch tensor to a PIL Image.

    Args:
    tensor (torch.Tensor): The image tensor to convert.

    Returns:
    PIL.Image: The image as a PIL Image.
    """
    tensor = tensor.squeeze(0)
    assert tensor.ndim == 3, f"Tensor has unsupported shape: {tensor.shape}"
    image_np = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image_np = (image_np * 255).astype(np.uint8)
    return Image.fromarray(image_np)


def clf_predict(
        model: nn.Module,
        test_dataloader,
        num_classes,
        threshold,
        device,
        amp,
        last_layer,
        cam_value,
        cam_save_path

):
    model.eval()

    predictions = []
    probabilities = []
    true_labels = []

    with torch.cuda.amp.autocast(enabled=amp):
        for batch in tqdm(test_dataloader, desc='Prediction round', unit='batch', leave=False):
            image, label, filename = batch['images'], batch['labels'], batch['filenames'][0]
            # move images and labels to the correct device
            images = image.to(device=device, dtype=torch.float32)
            pil_image = tensor2pil(image)

            # predict the class
            with torch.no_grad():
                outputs = model(images)

                if num_classes == 1:
                    # For binary classification, use sigmoid
                    predicted_prob = torch.sigmoid(outputs).cpu()
                    predicted = (predicted_prob > threshold).float()
                    probabilities.extend(predicted_prob.numpy()[:, 1])
                else:
                    # For multi-class classification, use softmax
                    softmax_outputs = F.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(softmax_outputs, 1)
                    _, predicted = confidence.cpu().numpy(), predicted.cpu().numpy()
                    probabilities.extend(softmax_outputs.cpu().numpy()[:, 1])

            # collect the predictions and true labels
            predictions.extend(predicted)
            true_labels.extend(label.cpu().numpy())

            # draw grad-cam
            if last_layer is not None:
                cam_value = int(predicted[0])
                print(cam_value)
                full_cam_save_path = f'{cam_save_path}{cam_value}_{label.cpu().numpy()[0]}'
                if not os.path.exists(full_cam_save_path):
                    os.makedirs(full_cam_save_path)
                draw_grad_cam(model=model, target_layers=[last_layer], cuda=True, pil_image=pil_image,
                              target_category=cam_value, save_path=f'{full_cam_save_path}/{filename}')

    return np.array(probabilities), predictions, true_labels


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # initial setting
    model_name = args.model_name
    load_path = args.load_model
    json_folder = args.input
    size = args.size
    output_folder = args.output
    viz = args.viz
    cam_value = args.cam_value
    threshold = args.threshold
    amp = args.amp

    # create dataset and data loaders
    test_dataset = ClassificationDatasetJson(json_folder, size=args.size)
    loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=True, **loader_args)
    classes = test_dataset.get_classes()

    # loading model and set last later
    model = None
    target_layer = None
    assert model_name in MODEL, "Model optional type are :{MODEL}, please choose available model_name"
    if model_name.upper() == 'EFFICIENT-NET':
        model = EfficientNetClassification(in_channels=3, num_classes=len(classes))
        if viz:
            target_layer = model.efficient_net._conv_head
    elif model_name.upper() == 'RES-NET':
        model = ResNetClassification(num_classes=len(classes))
        target_layer = model.resnet.layer4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model_name {model_name}')
    logging.info(f'Using device {device}')

    # loading pre-trained model
    model.to(device=device)
    state_dict = torch.load(load_path, map_location=device)
    if 'n_classes' in state_dict:
        del state_dict['n_classes']
    model.load_state_dict(state_dict)
    logging.info('Model loaded!')

    # initial matplotlib and save path
    std_mpl()
    output_path_final = f'{output_folder}/{model_name}/{now_time}/{now_h}'
    cam_save_path = f'{output_path_final}/CAM'
    os.makedirs(output_path_final, exist_ok=True)

    logging.info('Start Test!')
    probabilities, predictions, true_labels = clf_predict(
        model=model,
        test_dataloader=test_loader,
        num_classes=len(classes),
        threshold=threshold,
        device=device,
        amp=amp,
        last_layer=target_layer,
        cam_value=cam_value,
        cam_save_path=cam_save_path
    )

    # calculate and save the indicators
    acc = accuracy_score(true_labels, predictions, normalize=True)
    precision = precision_score(true_labels, predictions, average='micro')
    recall = recall_score(true_labels, predictions, average='micro')
    f1 = f1_score(true_labels, predictions, average='micro')
    ap = 0
    cf = confusion_matrix(true_labels, predictions)
    metrics = {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Average Precision': ap,
        'Confusion Matrix': cf.tolist()  # Convert Numpy to list
    }
    with open(f'{output_path_final}/metrics.txt', 'w') as file:
        for key, value in metrics.items():
            file.write(f'{key}: {value}\n')

    # viz the indicators
    class_labels = ['mel', 'nv']
    if viz:
        # draw confusion matrix
        draw_confusion_matrix(true_labels, predictions, classes, model_name)
        plt.savefig(f'{output_path_final}/CM.jpg')

        # draw roc
        draw_roc(true_labels, probabilities, model_name, class_labels)
        plt.savefig(f'{output_path_final}/ROC.jpg')


