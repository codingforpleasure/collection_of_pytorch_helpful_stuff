import os
from tqdm import tqdm
import math
from PIL import Image
import glob

if __name__ == '__main__':
    # files = glob.glob('/media/gil_diy/DEECBE15ECBDE843/Datasets_kaggle/diabetic-retinopathy-detection/train/*.jpeg')
    files = glob.glob('/media/gil_diy/DEECBE15ECBDE843/Datasets_kaggle/diabetic-retinopathy-detection/test/*.jpeg')

    new_width = 650

    for i in tqdm(range(len(files))):
        img = Image.open(files[i])
        width, height = img.size
        ratio = height / width
        if width > new_width:
            new_image = img.resize((new_width, math.ceil(ratio * new_width)))
        else:
            new_image = img

        # new_image.save(os.path.join(
        #     '/media/gil_diy/DEECBE15ECBDE843/Datasets_kaggle/diabetic-retinopathy-detection/train_650_size',
        #     os.path.basename(files[i])))
        new_image.save(os.path.join(
            '/media/gil_diy/DEECBE15ECBDE843/Datasets_kaggle/diabetic-retinopathy-detection/test_650_size',
            os.path.basename(files[i])))
