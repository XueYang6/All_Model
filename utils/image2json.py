import os
import json
import random


import logging
from tqdm import tqdm

import cv2 as cv
import numpy as np
from PIL import Image

from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)


def remap_mask_classes(mask, unique_values):
    """
        Remaps the mask's pixel values so that they match the given array of unique values.
    :param mask: numpy array of mask images
    :param unique_values: an array of unique pixel values that needs to be mapped to
    :return:  mask after remapping
    """
    # check mask_values for duplicate values
    assert len(unique_values) == len(set(unique_values)), "The mask_values contains duplicate values!"

    remapped_mask = np.copy(mask)
    sorted_unique_values = sorted(unique_values)  # mask sure unique values are sorted

    # calculate the new values that each pixel value should be mapped to
    for pixel_value in np.unique(mask):
        closest_value = min(sorted_unique_values, key=lambda x: abs(x - pixel_value))
        remapped_mask[mask == pixel_value] = closest_value
    return np.array(remapped_mask)


def map_unique_mask2categories(unique_mask):
    unique_mask_sorted = sorted(unique_mask)
    category_mapping = {mask: idx for idx, mask in enumerate(unique_mask_sorted)}
    return category_mapping


def extract_mask_coords(mask_data, unique_values):
    coords = {}
    for unique_value in unique_values:
        if unique_value != 0:  # exclude background
            polygon_coords = []
            # Find the one-dimensional index of the corresponding category
            binary_image = np.where(mask_data == unique_value, 1, 0).astype(np.uint8)
            contours, _ = cv.findContours(binary_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = cv.approxPolyDP(contour, epsilon=1.0, closed=True)
                polygon_coords.append(contour)
            coords[unique_value] = polygon_coords
    return coords


class ImageSegmentation2Json:
    def __init__(self, input_folder, output_folder, masks_folder, unique_values, size):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.masks_folder = masks_folder
        self.unique_values = unique_values
        self.json_data = []
        self.size = size

    def process_mask(self, filename):
        try:
            image_path = os.path.join(self.input_folder, filename)
            mask_path = os.path.join(self.masks_folder, filename)
            mask = Image.open(mask_path).resize(self.size)

            if mask.mode != 'L':
                mask = mask.convert('L')  # Convert to grayscale format
            mask_data = np.array(mask)
            remapped_mask = remap_mask_classes(mask_data, self.unique_values)
            mask = Image.fromarray(remapped_mask)
            mask.save(f'../datas/masks/{filename}.jpg')
            # extract mask  one-dimensional coordinates
            polygons_coords = extract_mask_coords(remapped_mask, self.unique_values)

            return {
                "id": filename,
                "polygons_coords": polygons_coords
            }
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            return None

    def convert_images2json(self):
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self.process_mask, filename) for filename in os.listdir(self.input_folder)
                       if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for future in tqdm(futures):
                self.json_data.append(future.result())

        json_path = self.output_folder + 'dataset.json'

        with open(json_path, 'w') as json_file:
            json.dump(self.json_data, json_file, indent=4)

        logging.info(f"The image is successfully converted to JSON \n "
                     f"save in: {self.output_folder} \n"
                     f"mask values: {self.unique_values}")


class MaskRCNNAnnotationGenerator:
    def __init__(self, image_dir, mask_dir, save_path, category_ids, mask_values):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.save_path = save_path
        self.category_ids = category_ids
        self.mask_values = mask_values
        self.mapped_category_id = map_unique_mask2categories(self.mask_values)
        self.images = []
        self.annotations = []
        self.ann_id = 0

    def generate_annotations(self):
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            [executor.submit(self._process_image, filename) for filename in os.listdir(self.image_dir)
             if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

        coco_format_json = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": [{"id": id, "name": name} for id, name in self.category_ids.items()]
        }

        with open(self.save_path, 'w') as f:
            json.dump(coco_format_json, f)

    def _process_image(self, filename):
        image_id = filename.split('.')[0]

        # Load image to get dimensions
        image = Image.open(os.path.join(self.image_dir, filename))
        image_info = {
            "id": image_id,
            "file_name": filename,
            "width": image.width,
            "height": image.height,
        }
        self.images.append(image_info)

        self._process_mask(image_id, filename)

    def _process_mask(self, image_id, filename):
        mask_filename = filename.replace('.png', '.jpg')  # replace .png to .jpg
        mask = Image.open(os.path.join(self.mask_dir, mask_filename))
        mask_np = remap_mask_classes(mask, self.mask_values)

        for category in self.mask_values:
            if category == 0:  # Skip background
                continue
            category_id = self.mapped_category_id[category]
            self._create_annotation(image_id, category, category_id, mask_np)

    def _create_annotation(self, image_id, category, category_id, mask_np):
        pos = np.where(mask_np == category)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        bbox = [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]

        mask = mask_np == category
        segmentation = []
        contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
        for contour in contours:
            contour = cv.approxPolyDP(contour, epsilon=1.0, closed=True)
            contour = contour.flatten().tolist()

            if len(contour) >= 6:
                segmentation.append(contour)

        annotation = {
            "id": image_id,
            "ann_id": self.ann_id,
            "category_id": int(category_id),
            "segmentation": segmentation,
            "bbox": bbox,
            "area": int(bbox[2] * bbox[3]),
        }
        self.annotations.append(annotation)
        self.ann_id += 1


# method of abandonment
def create_labels_file(input_folder, labels_file):
    labels = {}

    # Traverse all folders in the root directory
    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)

        # make sure it is a folder
        if os.path.isfile(folder_path):

            # Traverse all files in the folder
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    labels[filename] = folder_name

    with open(labels_file, 'w') as file:
        json.dump(labels, file, indent=4)


class ImageClassification2Json:
    def __init__(self, input_folder, output_folder, test_split=0.2):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.test_split = test_split
        self.train_json_data = []
        self.test_json_data = []
        self.class_index = self.create_class_index()

    def create_class_index(self):
        # create a dict to map class names with numbers
        categories = [d for d in os.listdir(self.input_folder) if os.path.isdir(os.path.join(self.input_folder, d))]

        return {category: index for index, category in enumerate(categories)}

    def split_data(self, file_list):
        # split data in train and test
        total_files = len(file_list)
        test_size = int(total_files * self.test_split)
        test_files = random.sample(file_list, test_size)
        train_files = [file for file in file_list if file not in test_files]
        return train_files, test_files

    def convert_images2json(self):
        train_class_info = {}
        test_class_info = {}

        # read every directory
        for folder_name, class_index in self.class_index.items():
            folder_path = os.path.join(self.input_folder, folder_name)
            all_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            train_files, test_files = self.split_data(all_files)

            # get number of images in each categorise
            train_class_info[class_index] = {
                'class_name': folder_name,
                'number_of_images': len(train_files)
            }

            test_class_info[class_index] = {
                'class_name': folder_name,
                'number_of_images': len(test_files)
            }

            # add images information
            for filename in train_files:
                self.train_json_data.append({
                    'filename': filename,
                    'image_path': os.path.join(folder_path, filename),
                    'label': class_index
                })

            for filename in test_files:
                self.test_json_data.append({
                    'filename': filename,
                    'image_path': os.path.join(folder_path, filename),
                    'label': class_index
                })

        self.save_json({'class_info': list(train_class_info.values()), 'images': self.train_json_data},
                       'Train_dataset.json')
        self.save_json({'class_info': list(test_class_info.values()), 'images': self.test_json_data},
                       'Test_dataset.json')

        logging.info(f"The image is successfully converted to json and split into train and test sets \n"
                     f"saved in: {self.output_folder} \n")

    def save_json(self, data, file_name):
        json_path = self.output_folder + file_name
        with open(json_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    # logging.info("Start test")
    # logging.info('Classification test')
    # images_path = 'E:/Datas/work/HairEffect/RawData/mel_vs_nv/train/all'
    # output_path = '../datasets/json/MelNv'
    # imageClf = ImageClassification2Json(images_path, output_path, test_split=0)
    # imageClf.convert_images2json()

    # logging.info('Segmentation test')
    # images_path = 'E:/Datas/work/HairEffect/SegmentData/ISIC2018_IMAGES'
    # masks_path = 'E:/Datas/work/HairEffect/SegmentData/ISIC2018_MASKS_RENAME'
    # output_path = '../datasets/images/Seg_ISIC2018'
    # imageSeg = ImageSegmentation2Json(images_path, output_path, masks_path, [0, 255], (1024, 1024))
    # imageSeg.convert_images2json()

    category_ids = {1: 'disease'}
    mask_values = [0, 255]
    generator = MaskRCNNAnnotationGenerator('E:/Datas/work/HairEffect/RawData/HAM10000/HAM10000_images',
                                            'E:/Datas/work/HairEffect/RawData/HAM10000/HAM10000_MASKS_RENAME',
                                            '../datasets/json/maskrcnn_annotation_HAM10000.json',
                                            category_ids,
                                            mask_values)
    generator.generate_annotations()

