
import csv
import logging
import numpy as np
import cv2 as cv

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit
from .cam_tools import GradCAM, show_cam_on_image
from PIL import Image

import torch
from torch.utils.data import Subset
import torchvision.transforms as transforms

dpi = 500


def std_mpl():
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['font.style'] = 'normal'

    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.grid'] = False

    legend = plt.legend()
    legend.set_frame_on(False)


def stratified_split(dataset, test_size, random_state=0):
    # Extract labels from the dataset
    labels = [y for _, y in dataset]

    # Create a stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, val_indices = next(sss.split(list(range(len(labels))), labels))

    # Create subsets for training and validation
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    return train_set, val_set


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, loss: bool = False):
        self.loss = loss
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.loss:
            score = -val_score
        else:
            score = val_score

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                logging.info(f"EarlyStopping counter: {self.counter}/{self.patience} \n "
                             f"best({self.best_score}), now({score})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


def save_indicators2csv(all_losses, name, location):
    file_name = f'{location}/{name}.csv'

    # Write the data in csv
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for epoch_loss in all_losses:
            writer.writerow(epoch_loss)

    logging.info(f'{name} is saved to {file_name}')


def draw_roc(true, proba, title, class_labels):
    # Check if `proba` is a 2D array and has more than one class
    if proba.ndim == 2 and proba.shape[1] > 1:
        # Calculating the ROC AUC score for multi-class
        auc = metrics.roc_auc_score(true, proba, multi_class='ovo')

        # Generate ROC curve values for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = proba.shape[1]
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(true, proba[:, i], pos_label=i)
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        # Plotting all ROC curves
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
        for i, color in zip(range(n_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(class_labels[i], roc_auc[i]))
    else:
        auc = metrics.roc_auc_score(true, proba)
        fpr, tpr, thresholds_test = metrics.roc_curve(true, proba)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=2000)
        ax.plot(fpr, tpr, color='black', lw=1.5, label='ROC curve (area = %0.4f)' % auc)
        ax.plot([0, 1], [0, 1], color='#d8d8d8', lw=1.5, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.spines['top'].set_color('none')  # 将顶部边框线颜色设置为透明
        ax.spines['right'].set_color('none')  # 将右侧边框线颜色设置为透明
        plt.legend(loc="lower right")


def draw_confusion_matrix(true, y_pre, display_labels, title='Confusion Matrix'):
    cm = confusion_matrix(true, y_pre, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig, ax = plt.subplots(dpi=dpi)
    disp.plot(cmap='Oranges', ax=ax)
    plt.gcf().subplots_adjust(left=0.2, bottom=0.2)
    plt.title(title)


def draw_grad_cam(model: torch.nn.Module, target_layers, cuda: bool, pil_image: Image, target_category: int, save_path: str):
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])

    image = pil_image.convert('RGB')
    image = np.array(image, dtype=np.uint8)
    image_tensor = data_transform(image)
    input_tensor = torch.unsqueeze(image_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=cuda)
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(image.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)

    # Create a color bar to represent relevance
    # Define a custom colormap from purple/blue to orange
    # Define a custom colormap from blue-purple to red-orange

    cmap_colors = [(0.6, 0, 1), (0, 0, 1), (0, 0.5, 1), (0, 1, .8), (0.8, 1, 0), (1, 0.5, 0), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('Custom', cmap_colors, N=100)

    heatmap = cmap(grayscale_cam)
    fig, ax = plt.subplots()
    plt.imshow(visualization)
    plt.axis('off')

    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])  # Adjust these values as needed
    cbar = plt.colorbar(ScalarMappable(cmap=cmap), cax=cbar_ax, orientation='vertical')
    cbar.set_label('Relevance', rotation=270, labelpad=15)
    cbar.ax.set_yticklabels([])  # Remove colorbar ticks
    # Define class labels (update with your own class labels)
    class_labels = ['nv', 'bkl', 'mel', 'akiec', 'bcc', 'vasc', 'df']
    # Add a title to the plot
    plt.title(f'{class_labels[target_category]}\'s Grad-CAM Map', fontsize=14)
    plt.savefig(save_path, dpi=500)
    logging.info('Save Cam')
    plt.close()


class ToHSV(object):
    """Convert a PIL image or tensor to HSV color space."""
    def __call__(self, image):
        return image.convert('HSV')


class ApplyCLAHE(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image):
        clahe = cv.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        image = np.array(image)
        if image.shape[-1] == 3:  # if RGB
            # apply CLAHE to each channel
            image = cv.cvtColor(image, cv.COLOR_RGB2LAB)  # turn to LAB
            image[..., 0] = clahe.apply(image[..., 0])  # Apply CLAHE only to luma channel
            image = cv.cvtColor(image, cv.COLOR_LAB2RGB)  # turn back to RGB
        else:
            image = clahe.apply(image)
        return Image.fromarray(image)

