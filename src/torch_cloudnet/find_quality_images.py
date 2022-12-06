from pathlib import Path
from tqdm import tqdm
from numpy import ndarray
import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage.io import imread
from sklearn.model_selection import train_test_split

from cs_6804_project.src.keras_cloudnet.utils import get_input_image_names

def _validate_test_ims(img: ndarray, num_pixel_thold: float = 0.8, intensity_thold: float = 20):
    """
    Hidden method that validates an image at a lower level.
    This is currently not being used as it takes ~ 30 mins
    (compared to the current check, which takes 5 mins). Might
    be a good idea to consider if we should use this or not
    :param img: Image to perform check on
    :type img: ndarray
    :param num_pixel_thold: % of pixels that should be below
                            intensity_thold
    :type num_pixel_thold: float
    :param intensity_thold: threshold for the intensity of an
                            image at a given location
    :type intensity_thold: float
    :return: Boolean indicating if the image passed the check
    :rtype: Boolean
    """
    nrow, ncol = img.shape
    nvals = nrow * ncol
    nwrong = 0
    for row in range(nrow):
        for col in range(ncol):
            nwrong += 1 if img[row][col] < intensity_thold else 0
    return (nwrong / nvals) < num_pixel_thold

def validate_test_ims(folder_path: Path, existing_nonempty: pd.DataFrame):
    """
    Function responsible for determining if images are valid or not
    :param folder_path: test folder path
    :type folder_path: Path
    :return:
    """
    blue_path = folder_path / 'train_blue_additional_to38cloud'
    red_path = folder_path / 'train_red_additional_to38cloud'
    green_path = folder_path / 'train_green_additional_to38cloud'
    nir_path = folder_path / 'train_nir_additional_to38cloud'

    blue_files = os.listdir(blue_path)
    red_files = os.listdir(red_path)
    green_files = os.listdir(green_path)
    nir_files = os.listdir(nir_path)

    valid_list = []
    im_sums = []

    for idx in tqdm(range(len(blue_files)), total=len(blue_files)):
        blue = blue_files[idx]
        patch_name = blue.replace('blue_', '')[:-4]
        if (patch_name not in existing_nonempty['name'].values):
            continue
        blue_sum = sum(sum(imread(blue_path / blue)))
        red_sum = sum(sum(imread(red_path / red_files[idx])))
        green_sum = sum(sum(imread(green_path / green_files[idx])))
        nir_sum = sum(sum(imread(nir_path / nir_files[idx])))
        intensities = blue_sum + red_sum + nir_sum + green_sum
        valid_check = intensities > 100
        #if 0 in imread(blue_path / blue):
            #plt.imshow(imread(blue_path / blue))
            #plt.show()
        im_sums.append(intensities)
        if valid_check:
            valid_list.append(patch_name)

    valid_test_ims = pd.DataFrame(data={"name": valid_list})
    valid_test_ims.to_csv(folder_path / 'updated_evaluation_testing_patches_nonempty.csv', index=False)


if __name__ == "__main__":
    GLOBAL_PATH = Path('../../data')
    TRAIN_FOLDER = GLOBAL_PATH / '38-Cloud_training'
    TEST_FOLDER = GLOBAL_PATH / '38-Cloud_test'
    LOG_DIR = "you must choose a path wisely, Daniel"
    in_rows = 192
    in_cols = 192
    num_of_channels = 4
    num_of_classes = 1
    starting_learning_rate = 1e-4
    end_learning_rate = 1e-8
    max_num_epochs = 2000  # just a huge number. The actual training should not be limited by this value
    val_ratio = 0.2
    patience = 15
    decay_factor = 0.7
    batch_sz = 12
    max_bit = 65535  # maximum gray level in landsat 8 images

    validate_test_ims(folder_path=TEST_FOLDER)

    # getting input images names
    train_patches_csv_name = 'training_patches_38-cloud_nonempty.csv'
    test_patches_csv_name = 'testing_patches_38-cloud_nonempty.csv'
    df_train_img = pd.read_csv(TRAIN_FOLDER / train_patches_csv_name)
    df_test_img = pd.read_csv(TEST_FOLDER / test_patches_csv_name)

    test_img, test_msk = get_input_image_names(df_test_img, TEST_FOLDER, if_train=True)
    train_img, train_msk = get_input_image_names(df_train_img, TRAIN_FOLDER, if_train=True)
    # train_img = train_img[:100]
    # train_msk = train_msk[:100]

    # Split data into training and validation
    train_img_split, val_img_split, train_msk_split, val_msk_split = train_test_split(
        train_img, train_msk, test_size=val_ratio, random_state=42, shuffle=True
    )
