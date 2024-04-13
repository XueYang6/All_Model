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
from utils.data_loading import remap_mask_classes
from .metrics import proba_metrics, euclidean_distance

dpi = 500


def std_mpl():
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['font.style'] = 'normal'

    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.grid'] = False


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
    if proba.ndim == 2 and proba.shape[0] > 1:
        # Calculating the ROC AUC score for multi-class
        metrics.roc_auc_score(true, proba, multi_class='ovo')

        # Generate ROC curve values for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = proba.shape[0]
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
        ax.plot(fpr, tpr, color='orange', lw=1.5, label='ROC curve (area = %0.4f)' % auc)
        ax.plot([0, 1], [0, 1], color='#d8d8d8', lw=1.5, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.spines['top'].set_color('none')  # 将顶部边框线颜色设置为透明
        ax.spines['right'].set_color('none')  # 将右侧边框线颜色设置为透明
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        plt.legend(loc="lower right", frameon=False)

    return fpr, tpr


def draw_confusion_matrix(true, y_pre, display_labels, title='Confusion Matrix'):
    cm = confusion_matrix(true, y_pre, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig, ax = plt.subplots(dpi=dpi)
    disp.plot(cmap='Oranges', ax=ax)
    plt.gcf().subplots_adjust(left=0.2, bottom=0.2)
    plt.title(title)


def get_cam_value(model: torch.nn.Module, target_layers, cuda: bool, pil_image: Image, target_category: int):
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])
    image = pil_image.convert('RGB')
    image = np.array(image, dtype=np.uint8)
    image_tensor = data_transform(image)
    input_tensor = torch.unsqueeze(image_tensor, dim=0)
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=cuda)
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    return grayscale_cam[0, :]


def draw_grad_cam(model: torch.nn.Module, target_layers, cuda: bool, pil_image: Image, target_category: int,
                  save_path: str):
    image = pil_image.convert('RGB')
    image = np.array(image, dtype=np.uint8)

    grayscale_cam = get_cam_value(model, target_layers, cuda, pil_image, target_category)
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
    # Add a title to the plot
    plt.title(f' Grad-CAM Map', fontsize=14)
    plt.savefig(save_path, dpi=500)
    logging.info('Save Cam')
    plt.close()


def calculate_mask_centroid(mask: np.array):
    assert mask.ndim == 2, "Mask must be 2D"

    non_zero_position = np.argwhere(mask > 0)

    centroid = np.mean(non_zero_position, axis=0)

    return tuple(np.round(centroid).astype(int))


def get_cam_salient_center(cam):
    max_val = np.max(cam)
    max_positions = np.argwhere(cam == max_val)
    max_position = tuple(max_positions[0])

    return max_position, max_val


def get_mask_max_diameter(true_mask):
    """Calculate the size of the lesion's bounding box from the mask"""
    rows = np.any(true_mask, axis=1)
    cols = np.any(true_mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    return max(width, height)


def get_cam_metrics(model: torch.nn.Module, target_layers, cuda: bool, pil_image: Image, target_category: int,
                    true_mask, save_path=None):
    grayscale_cam = get_cam_value(model, target_layers, cuda, pil_image, target_category)
    max_position, max_val = get_cam_salient_center(grayscale_cam)
    centroid = calculate_mask_centroid(true_mask)
    dx, dy, distance = euclidean_distance(max_position, centroid)

    max_diameter = get_mask_max_diameter(true_mask)
    max_radius = max_diameter / 2

    if save_path is not None:
        cam = np.uint8(255 * grayscale_cam)
        cv.imwrite(f'{save_path}', cam)

    inf1, inf2 = proba_metrics(grayscale_cam, true_mask)

    inf3 = 1 - (distance / max_radius)
    inf = (inf1 + inf2 + inf3) / 3
    return inf1, inf2, inf3, inf


class ToHSV(object):
    """Convert a PIL image or tensor to HSV color space."""

    def __call__(self, image):
        return image.convert('HSV')


class ApplyCLAHE(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), channel='RGB', enhance: int = 0):
        self.color_spaces = ['RGB', 'LAB', 'HSV']

        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        assert channel.upper() in self.color_spaces, f'choose channel in {self.color_spaces}, but give {channel}'
        self.channel = channel.upper()
        self.enhance = enhance

    def convert_color_space(self, image):
        if self.channel == 'LAB':
            return cv.cvtColor(image, cv.COLOR_RGB2LAB)
        if self.channel == 'HSV':
            return cv.cvtColor(image, cv.COLOR_RGB2HSV)
        return image

    def convert_back2rgb(self, image):
        if self.channel == 'LAB':
            return cv.cvtColor(image, cv.COLOR_LAB2RGB)
        if self.channel == 'HSV':
            return cv.cvtColor(image, cv.COLOR_HSV2RGB)
        return image

    def __call__(self, image):
        clahe = cv.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        image_np = np.array(image)

        if image_np.ndim == 3 and image_np.shape[2] == 3:  # if 3 channel
            # apply CLAHE enhance channel
            converted_image = self.convert_color_space(image_np)
            converted_image[..., self.enhance] = clahe.apply(converted_image[..., self.enhance])  # Apply CLAHE
            image_np = self.convert_back2rgb(converted_image)
        elif image_np.ndim == 2 or (image_np.ndim == 3 and image_np.shape[2] == 1):  # Grayscale
            image_np = clahe.apply(image_np)
        else:

            Image.fromarray(image_np).convert('RGB').save('1.jpg')
            raise ValueError("Unsupported image format for CLAHE")
        return Image.fromarray(image_np)


def show_transformed_images(image_path, transform):
    # original image
    original_image = Image.open(image_path).convert('RGB')
    # use transform
    transformed_image = transform(original_image)

    # back to PILImage
    transformed_image_pil = transforms.ToPILImage()(transformed_image)

    # to show image
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(transformed_image_pil)
    ax[1].set_title("Transformed Image")
    ax[1].axis('off')

    plt.show()


def mask2image(mask_path, image_path, save_path, size):
    image = cv.imread(image_path)
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

    image_resized = cv.resize(image, size, interpolation=cv.INTER_AREA)
    mask_resized = cv.resize(mask, size, interpolation=cv.INTER_AREA)

    mask_np = remap_mask_classes(mask=mask_resized, unique_values=[0, 255])

    binary_mask_3channel = cv.cvtColor(mask_np, cv.COLOR_GRAY2BGR)
    result = cv.bitwise_and(image_resized, binary_mask_3channel)
    cv.imwrite(save_path, result)
