import numpy as np
from torch.utils.data import Dataset
from skimage.io import imread
from skimage.transform import resize
from torchvision.transforms import ToTensor
from cs_6804_project.src.keras_cloudnet.augmentation import *


class CloudDataset(Dataset):
    def __init__(self, train_files, target_files, img_rows, img_cols, max_bit, transform=True):
        self.train_files = train_files
        self.target_files = target_files
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.max_bit = max_bit
        self.transform = transform

    def __len__(self):
        return len(self.target_files)

    def __getitem__(self, idx):
        # Get input images
        image_red = imread(self.train_files[idx][0])
        image_green = imread(self.train_files[idx][1])
        image_blue = imread(self.train_files[idx][2])
        image_nir = imread(self.train_files[idx][3])
        images = np.stack((image_red, image_green, image_blue, image_nir), axis=-1).astype('int32')
        images = resize(images, (self.img_rows, self.img_cols),
                        order=0, preserve_range=True, mode='symmetric', anti_aliasing=False)
        # Get target image
        target = imread(self.target_files[idx])
        target = resize(target, (self.img_rows, self.img_cols),
                        order=0, preserve_range=True, mode='symmetric', anti_aliasing=False)

        # get file name to know which file corresponds to which predictions

        # Perform image augmentation
        if self.transform:
            images, target = self.transform_data(images, target)

        # file name without color
        normalized_fname = '_'.join(self.train_files[idx][0].name.split('_')[2:])

        # Switch to CHW format and convert to Dataset
        return ToTensor()(images), ToTensor()(target), [normalized_fname]

    def transform_data(self, images, target):
        rnd_flip = np.random.randint(2, dtype=int)
        rnd_rotate_clk = np.random.randint(2, dtype=int)
        rnd_rotate_cclk = np.random.randint(2, dtype=int)
        rnd_zoom = np.random.randint(2, dtype=int)

        if rnd_flip == 1:
            images, target = flipping_img_and_msk(images, target)

        if rnd_rotate_clk == 1:
            images, target = rotate_clk_img_and_msk(images, target)

        if rnd_rotate_cclk == 1:
            images, target = rotate_cclk_img_and_msk(images, target)

        if rnd_zoom == 1:
            images, target = zoom_img_and_msk(images, target)

        # target /= 255
        target = np.divide(target, 255)
        # images /= self.max_bit
        images = np.divide(images, self.max_bit)
        return images, target
