import argparse
import logging
import os

import sklearn
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
import matplotlib as mpl

from utils.utils import draw_grad_cam, draw_roc, draw_confusion_matrix, std_mpl

from models.seg.UNet.unet_model import UNet
from utils.plot import plot_img_and_mask

std_mpl()

def predict_img(net,
                full_img,
                device,
                size=(128, 128),
                out_threshold=0.7):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, size, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()

        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold
            proba = torch.sigmoid(output)

    return mask[0].long().squeeze().numpy(), proba[0].float().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks_f from input images_0')
    parser.add_argument('--model_name', '-m', metavar='FILE',
                        default='best/best.pth',
                        help='Specify the file in which the model_name is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        default='crop_datas/crop96/patient_images2left_clahe', help='Filenames of input images_0')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+',
                        default='predict/predict_images/all', help='Filenames of output images_0')
    parser.add_argument('--roc', '-roc', metavar='ROC',
                        default='crop_datas/crop96/patient_masks2left_2class', help='true_mask_address')
    parser.add_argument('--viz', '-v', action='store_true',
                        default='predict/visualization/all', help='Visualize the images_0 as they are processed')
    parser.add_argument('--grad-cam-value', '-gv', type=int,
                        default=0, help='saliency map values 2 means (1 and 2)')
    parser.add_argument('--grad-cam-path', '-gp', action='store_true',
                        default='predict/cam/all', help='saliency map save path ')
    parser.add_argument('--mask-threshold', '-t', type=float,
                        default=0.4, help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--size', '-s', type=tuple,
                        default=(128, 128), help='Scale factor for the input images_0')
    parser.add_argument('--bilinear', action='store_true',
                        default=False, help='Use bilinear upsampling')
    parser.add_argument('--n_classes', '-c', type=int,
                        default=1, help='Number of n_classes')

    args = parser.parse_args()
    return args


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_folder = args.input
    out_folder = get_output_filenames(args)
    true_mask_folder = args.roc

    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(args.viz, exist_ok=True)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model_name {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    out_image_path = None
    viz_save_path = None
    cam_save_path = None

    all_true = []
    all_proba = []
    all_predict = []

    images_path = os.listdir(in_folder)
    for image_path in tqdm(images_path):
        full_image_path = f"{in_folder}/{image_path}"
        image_name = image_path.split('.')[0]

        logging.info(f'Predicting image {full_image_path} ...')

        image = Image.open(full_image_path)

        mask, proba = predict_img(
            net=net,
            full_img=image,
            size=args.size,
            out_threshold=args.mask_threshold,
            device=device
        )

        full_mask_path = f"{true_mask_folder}/{image_path}"
        true_mask = np.asarray(Image.open(full_mask_path))
        true_mask = np.copy(true_mask)
        true_mask[true_mask < 50] = 0
        true_mask[true_mask >= 50] = 255
        all_true.append(true_mask / 255.0)
        all_predict.append(mask)
        all_proba.append(proba)

        out_image_path = f"{out_folder}/{image_path}"

        result = mask_to_image(mask, mask_values)
        result.save(out_image_path)
        logging.info(f'Mask saved to {out_image_path}')

        # save viz and cam
        viz_save_path = f"{args.viz}/viz-{image_path}"

        plot_img_and_mask(image=image, predict_mask=result, true_mask=true_mask, colors=[[255, 255, 0]], save_path=f'{viz_save_path}')
        if args.classes == 3:
            os.makedirs(f"{args.grad_cam_path}/cam{args.grad_cam_value-1}", exist_ok=True)
            os.makedirs(f"{args.grad_cam_path}/cam{args.grad_cam_value}", exist_ok=True)
            cam_save_path2 = f"{args.grad_cam_path}/cam{args.grad_cam_value}/{image_path}"
            cam_save_path1 = f"{args.grad_cam_path}/cam{args.grad_cam_value-1}/{image_path}"

            draw_grad_cam(model=net, target_layers=[net.up4], cuda=True, image_path=full_image_path,
                          target_category=args.grad_cam_value, save_path=f'{cam_save_path2}')
            draw_grad_cam(model=net, target_layers=[net.up4], cuda=True, image_path=full_image_path,
                          target_category=(args.grad_cam_value - 1), save_path=f'{cam_save_path1}')
        elif args.classes == 1:
            os.makedirs(f"{args.grad_cam_path}/cam{args.grad_cam_value}", exist_ok=True)
            cam_save_path = f"{args.grad_cam_path}/cam{args.grad_cam_value}/{image_path}"
            draw_grad_cam(model=net, target_layers=[net.up4], cuda=True, image_path=full_image_path,
                          target_category=args.grad_cam_value, save_path=f'{cam_save_path}')

    logging.info(f'Finish all oh the image,'
                 f' output save in {out_image_path}, '
                 f'visualization save in {viz_save_path}, '
                 f'Grad-CAM save in {cam_save_path}')

    if true_mask_folder is not None:
        true = np.array(all_true).reshape(-1)
        proba = np.array(all_proba).reshape(-1)
        predict = np.array(all_predict).reshape(-1)

        draw_roc(true, proba, 'ROC')
        plt.savefig("roc_left.jpg")
        plt.close()

        draw_confusion_matrix(true, predict, ['Non-Cochlea', 'Cochlea'])
        plt.savefig("CM_left.jpg")
        plt.close()

