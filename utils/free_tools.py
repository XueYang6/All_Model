import os
import shutil


def move_datas(target_path, aim_path, mid_path=None):
    ids = os.listdir(target_path)
    print(ids)
    for id in ids:
        file_name = id.split('.')[0] + '.bif'
        source_file = os.path.join(mid_path, file_name)
        destination_file = os.path.join(aim_path, file_name)

        shutil.copy(source_file, destination_file)


if __name__ == "__main__":

    path = "E:/Datas/work/HairEffect/RawData/HAM10000/HAM10000_MASKS_RENAME"
    ids = os.listdir(path)
    for id in ids:
        newname = id.split('_segmentation')[0] + '.jpg'
        file_path = f'{path}/{id}'
        new_path = f'{path}/{newname}'

        if os.path.exists(new_path):
            os.remove(new_path)
        os.rename(file_path, new_path)

