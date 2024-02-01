import os
import pandas as pd
import json


class CSV2JSON:
    def __init__(self, csv_file, image_folder, output_folder):
        self.df = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.output_folder = output_folder

        self.class_index = {dx: index for index, dx in enumerate(self.df['dx'].unique())}
        self.json_data = []

    def convert2json(self):
        class_info = {}

        for _, row in self.df.iterrows():
            image_id = row['image_id']
            class_name = row['dx']
            class_idx = self.class_index[class_name]
            filename = f'{image_id}.jpg'
            self.json_data.append({
                'filename': filename,
                'image_path': os.path.join(self.image_folder, filename),
                'label': class_idx
            })

            # update class info
            if class_idx not in class_info:
                class_info[class_idx] = {
                    'class_name': class_name,
                    'number_of_images': 0
                }

            class_info[class_idx]['number_of_images'] += 1

        # Save JSON data to files
        with open(f'{self.output_folder}_dataset.json', 'w') as file:
            json.dump({'classes': self.class_index, 'class_info': list(class_info.values()), 'images': self.json_data}, file, indent=4)


if __name__ == "__main__":
    image_path = 'E:/Datas/work/HairEffect/RawData/HAM10000/ISIC2018_Task3_Test_Images'
    csv_path = 'E:/Datas/work/HairEffect/RawData/HAM10000/ISIC2018_Task3_Test_GroundTruth.csv'
    out_path = '../datas/json_data/ISIC2018TEST'
    generator = CSV2JSON(csv_path, image_path, out_path)
    generator.convert2json()
