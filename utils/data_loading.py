import json
import logging
from tqdm import tqdm

import os
import cv2 as cv
from PIL import Image
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from .image2json import remap_mask_classes


import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, pil_to_tensor, to_pil_image

logging.basicConfig(level=logging.INFO)


def one_dim2dim_two_mask(mask_coords, size):
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    height, width = size
    for label, coords in mask_coords.items():
        for coord in coords:
            x = coord % width
            y = coord // width
            mask[y, x] = label

    return Image.fromarray(mask)


def unique_mask_values_json(json_data: json):
    unique_values = set()
    unique_values.add(0)

    for item in json_data:
        mask_coords = item['mask_coords']
        for value in mask_coords.keys():
            unique_values.add(int(value))

    return sorted(unique_values)


def preprocess(pil_image: Image, size: tuple, is_mask: bool, mask_values=None):
    if mask_values is None:
        mask_values = [0, 255]
    if is_mask:
        assert (mask_values is not None), f"If it is mask, unique mask values must be given"

    new_w, new_h = size
    assert new_w > 0 and new_h > 0, f"size({size}) must lager than 0"

    # resize image with BICUBIC and mask with NEAREST
    pil_image = pil_image.resize((new_w, new_h), resample=Image.NEAREST if is_mask else Image.BICUBIC)
    image = np.asarray(pil_image)

    # Narrow the range of 0-255 to 0-1 to enhance segmentation feature performance
    if is_mask:
        mask = np.zeros((new_h, new_w), dtype=np.int64)
        for i, v in enumerate(mask_values):
            if mask.ndim == 2:
                mask[image == v] = i
            else:
                mask[(image == v).all(-1)] = i
        return mask

    else:
        if image.ndim == 2:
            image = np.stack((image, image, image), axis=-1)
        image = image.transpose((2, 0, 1))

        if (image > 1).any():
            image = image / 255.0
        return image


class SegmentDatasetJson(Dataset):
    def __init__(self, image_dir, json_dir: str, size: tuple = (512, 512)):
        assert 0 < size[0], "size must > 0"
        self.size = size

        with open(json_dir, 'r') as json_file:
            self.json_data = json.load(json_file)

        self.image_dir = image_dir
        self.image_info = {d['id']: d for d in self.json_data['images']}
        self.annotations = self.json_data['annotations']

        self.classes = {c['id']: c['name'] for c in self.json_data['categories']}

        logging.info(f"Loading json file successfully!")
        logging.info(f'Create dataset with {len(self.image_info)}')
        logging.info(f"Classes: {self.classes}")

    def __getitem__(self, idx):
        item_id = list(self.image_info.keys())[idx]
        image_data = self.image_info[item_id]
        annotations = [ann for ann in self.annotations if ann['id'] == item_id]

        # Load image
        image_path = os.path.join(self.image_dir, image_data['file_name'])
        image = Image.open(image_path).convert('RGB')
        image = preprocess(image, is_mask=False, size=self.size, mask_values=None)

        # Load masks, labels and boxes
        masks = []
        labels = []
        boxes = []

        for ann in annotations:
            # Create mask from polygon
            mask = Image.new('L', (image_data['width'], image_data['height']))
            mask_np = np.array(mask)
            for segment in ann['segmentation']:
                mask_np = cv.fillPoly(mask_np, [np.array(segment).reshape(-1, 1, 2)], 1)

            unique_values = np.unique(mask_np)
            mask_pil = Image.fromarray(mask_np)
            processed_mask = preprocess(mask_pil, size=self.size, is_mask=True, mask_values=unique_values)
            masks.append(processed_mask)

            # reset bbox
            original_width, original_height = image_data['width'], image_data['height']
            target_width, target_height = self.size
            scale_x, scale_y = target_width / original_width, target_height / original_height
            x_min, y_min, width, height = ann['bbox']
            new_x_min, new_y_min = x_min * scale_x, y_min * scale_y
            new_width, new_height = width * scale_x, height * scale_y
            new_x_max, new_y_max = new_x_min + new_width, new_y_min + new_height
            ann['bbox'] = [new_x_min, new_y_min, new_x_max, new_y_max]
            boxes.append(ann['bbox'])

            labels.append(ann['category_id'])

        masks_combined = np.stack(masks, axis=0)

        # Convert everything to PyTorch tensors
        masks = torch.tensor(masks_combined).long().contiguous()
        labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (masks.sum((1, 2))).float()
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        # Make sure target is a dictionary and contains the required keys

        return {
            "image": torch.as_tensor(image.copy()).float().contiguous(),
            "target": target
        }

    def __len__(self):
        return len(self.image_info)


class SegmentationDatasetDirectory(Dataset):
    def __init__(self, image_dir, mask_dir, size: tuple = (512, 512), mask_values=None, transform=None):
        self.images_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        assert mask_values is not None, "mask values must not None"
        assert 0 < size[0], "size must > 0"
        self.mask_values = mask_values
        self.size = size

        self.ids = [splitext(file)[0] for file in listdir(image_dir)
                    if isfile(join(image_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file in {image_dir}')

        logging.info(f'Create dataset with {len(self.ids)}')

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + '.*'))
        image_file = list(self.images_dir.glob(name + '.*'))
        assert len(image_file) == 1, f'Either no images or multiple images_0 found for the ID{name}: {image_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks_f found for the ID{name}: {mask_file}'

        image = Image.open(image_file[0])
        mask = Image.open(mask_file[0]).resize(self.size)
        remapped_mask = remap_mask_classes(np.array(mask), self.mask_values)
        mask = Image.fromarray(remapped_mask)

        if self.transform:
            image = self.transform(image)
        mask = preprocess(mask, self.size, True, self.mask_values)

        return {
            'images': image.float().contiguous(),
            'masks': torch.as_tensor(mask.copy()).long().contiguous()
        }

    def __len__(self):
        return len(self.ids)


class ClassificationDatasetJson(Dataset):
    def __init__(self, json_dir: str, size: tuple = (512, 512), transform=None):
        assert 0 < size[0], "size must > 0"
        self.size = size
        self.transform = transform

        with open(json_dir, 'r') as json_file:
            data = json.load(json_file)
            self.json_data = data['images']
            self.classes = [info['class_name'] for info in data.get('class_info', [])]

        logging.info(f'Loading json file successfully!')

    def __getitem__(self, idx):
        item = self.json_data[idx]
        image_path = item['image_path']

        label = item['label']
        image = Image.open(image_path)
        filename = item['filename']

        if self.transform:
            image = self.transform(image)

        return {
            'images': image.float().contiguous(),
            'labels': torch.tensor(label, dtype=torch.int64),
            'filenames': filename
        }

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.json_data)


def merge_masks(masks):
    """
    Merge masks

    (torch.Tensor): Mask tensor with shape [N, 1, H, W], where N is the number of detected objects.
    threshold (float): Barbarization threshold
.
    Returns
        torch.Tensor: The merged mask has shape [1, H, W].
    """
    if masks.size(0) == 0:
        # Handle the case of empty tensors, such as returning an empty mask
        return torch.zeros((1, masks.size(2), masks.size(3)))
    combined_mask = torch.max(masks, dim=0)[0]  # Take the maximum merge mask
    return combined_mask


if __name__ == "__main__":
    logging.info('Start Test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # logging.info('Classification Test')
    # json_path = '../datas/json_data/ClfWorktrain_dataset.json'
    # clf_dataset = ClassificationDatasetJson(json_path, (256, 256))
    # loader_args = dict(batch_size=8, num_workers=os.cpu_count(), pin_memory=True)
    # data_loader = DataLoader(clf_dataset, shuffle=True, **loader_args)
    # for batch in tqdm(data_loader):
    #     images, labels = batch['images'], batch['labels']

    # logging.info('Segmentation Test')
    # image_path = 'E:/Datas/work/HairEffect/SegmentData/ISIC2018_IMAGES'
    # mask_path = 'E:/Datas/work/HairEffect/SegmentData/ISIC2018_MASKS_RENAME'
    # seg_dataset = SegmentationDatasetDirectory(image_path, mask_path, (256, 256), [0, 255])
    # loader_args = dict(batch_size=8, num_workers=os.cpu_count(), pin_memory=True)
    # data_loader = DataLoader(seg_dataset, shuffle=True, **loader_args)
    # for batch in tqdm(data_loader):
    #     images, true_masks = batch['images'], batch['masks']

    logging.info('Segmentation Test2')
    annotation_path = '../datas/json_data/maskrcnn_annotation.json'
    image_path = 'E:/Datas/work/HairEffect/SegmentData/ISIC2018_IMAGES'
    seg_dataset = SegmentDatasetJson(image_path, annotation_path, (256, 256))

    # Split into train / validation partitions
    val_percent = 20 / 100
    n_val = int(len(seg_dataset) * val_percent)
    n_train = len(seg_dataset) - n_val
    train_set, val_set = random_split(seg_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # Create dataloader os.cpu_count()
    loader_args = dict(batch_size=1, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    for batch_idx, batch in enumerate(
            tqdm(val_loader, total=len(val_loader), desc='Validation round', unit='batch', leave=False)):
        images, target_dict = batch['image'], batch['target']
        masks, labels = target_dict['masks'], target_dict['labels']

        for i in range(images.shape[0]):
            image = images[i]
            image_pil = to_pil_image(image)

            merge_mask = merge_masks(masks[i])
            mask_pil = to_pil_image(merge_mask.byte())

            # 确保掩码尺寸与图像相匹配并且模式为 'L'
            mask_pil = mask_pil.resize(image_pil.size)
            mask_pil = mask_pil.convert("L")
            mask_pil = mask_pil.point(lambda p: p * 255)

            image_pil.paste(mask_pil, (0, 0), mask_pil)
            image_pil.save(f"check/{labels}_{i}.jpg")

