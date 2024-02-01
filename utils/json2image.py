import logging
import os
import json
from tqdm import tqdm
from PIL import Image


def json2image_folder(json_dir, save_path):

    with open(json_dir, 'r') as json_file:
        json_data = json.load(json_file)
        images_data = json_data['images']

        # get n_classes
        classes = [info['class_name'] for info in json_data.get('class_info', [])]
        print(f'classes: {classes}')

        for image_data in tqdm(images_data):
            image_path = image_data['image_path']
            image_name = image_data['filename']
            image_class = image_data['label']

            # save image
            try:
                image = Image.open(image_path)
            except IOError:
                print(f"Could not open or save image {image_name} from path {image_path}")

            image_save_path = os.path.join(save_path, classes[image_class])
            # ensure the save path exists, if not create it
            if not os.path.exists(image_save_path):
                logging.info('The save path not exist, create one')
                os.makedirs(image_save_path)

            image.save(f'{image_save_path}/{image_name}')


if __name__ == '__main__':
    json_dir = '../datas/json_data/ClfWorktest_dataset.json'
    save_dir = '../datas/images/test_images'
    json2image_folder(json_dir, save_dir)
