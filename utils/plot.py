import logging
from PIL import Image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import std_mpl
std_mpl()


def plot_img_and_mask(image, predict_mask, true_mask=None, save_path=None, colors=None):
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
        image_rgb = pil_image.convert('RGB')
        image = np.array(image_rgb)

    else:
        image_rgb = image.convert('RGB')
        image = np.array(image_rgb)

    if not isinstance(predict_mask, np.ndarray):
        predict_mask = np.array(predict_mask)

    if not isinstance(true_mask, np.ndarray) & (true_mask is not None):
        true_mask = np.array(true_mask)

    if colors is None:
        colors = [[0, 255, 0],
                  [255, 0, 0],
                  [0, 0, 255]]

    mask_values = np.unique(predict_mask)
    assert mask_values.all() == np.unique(true_mask).all(), f'Predict mask values must equal to true mask values, but give {np.unique(predict_mask)}, {np.unique(true_mask),}'
    assert len(mask_values) <= 4, f"It only ready for classes <= 4 (including background) but predict_mask give ({mask_values})"

    n_clos = 3 if true_mask is None else 5

    fig, ax = plt.subplots(1, n_clos, figsize=(12, 4))
    ax[0].set_title('Input images')
    ax[0].imshow(image)
    ax[0].axis('off')

    image_with_mask = np.copy(image)
    image_with_t_mask = np.copy(image)
    final_mask = np.zeros_like(image)
    final_t_mask = np.zeros_like(image)

    for i in range(1, len(mask_values)):
        mask = np.where(predict_mask == mask_values[i], 1, 0)
        mask_colorful = np.zeros_like(image)
        mask_colorful[mask.astype(bool)] = colors[i - 1]
        final_mask = cv.addWeighted(final_mask, 0.5, mask_colorful, 0.5, 0)
        if true_mask is not None:
            t_mask = np.where(true_mask == mask_values[i], 1, 0)
            t_mask_colorful = np.zeros_like(image)
            t_mask_colorful[t_mask.astype(bool)] = colors[i - 1]
            final_t_mask = cv.addWeighted(final_t_mask, 0.5, t_mask_colorful, 0.5, 0)

    image_with_mask = cv.addWeighted(image_with_mask, 1, final_mask, 0.8, 0)
    ax[1].set_title(f'final mask')
    ax[1].imshow(final_mask)
    ax[1].axis('off')

    ax[2].set_title(f'images with mask')
    ax[2].imshow(image_with_mask)
    ax[2].axis('off')

    if true_mask is not None:
        ax[3].set_title(f'final mask')
        ax[3].imshow(final_t_mask)
        ax[3].axis('off')

        image_with_t_mask = cv.addWeighted(image_with_t_mask, 1, final_t_mask, 0.8, 0)
        ax[4].set_title(f'images with true mask')
        ax[4].imshow(image_with_t_mask)
        ax[4].axis('off')

    plt.xticks([]), plt.yticks([])
    plt.savefig(save_path, dpi=500)
    logging.info('Save plot')
    plt.close()


def compare_images(image_pairs, titles):

    """
    Display multiple pairs of original and enhanced images_0 side by side for comparison.

    Parameters:
        image_pairs (list): A list of images pairs, each pair is a tuple (original_image, enhanced_image).
        titles (list): A list of titles for each pair of images_0.
    """

    num_pairs = len(image_pairs)
    assert len(titles) == num_pairs,\
        f"Number of titles({len(titles)}) must match the number of images pairs({num_pairs})"

    plt.figure(figsize=(15, 5 * num_pairs))

    for i in range(num_pairs):
        original_image, enhanced_imaged = image_pairs[i]
        title = titles[i]
        fig, axes = plt.subplots(num_pairs, 2)

        # Display original images
        axes[0].imshow(cv.cvtColor(original_image, cv.COLOR_BGR2RGB))
        axes[0].set_title(f'{title} - Original')
        axes[0].axis('off')

        # Display the enhanced images
        axes[1].imshow(enhanced_imaged, cmap='gray')
        axes[1].set_title(f'{title} - Enhanced')
        axes[1].axis('off')

    plt.tight_layout()
    plt.show()


