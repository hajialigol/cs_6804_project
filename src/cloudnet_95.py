from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd
import os


def read_cloudnet(test_eval_df: pd.DataFrame, cloudnet_95_path: Path):
    test_eval_fnames = set(test_eval_df['name'].to_list())

    blue_path = cloudnet_95_path / 'train_blue_additional_to38cloud'
    red_path = cloudnet_95_path / 'train_red_additional_to38cloud'
    green_path = cloudnet_95_path / 'train_green_additional_to38cloud'
    nir_path = cloudnet_95_path / 'train_nir_additional_to38cloud'
    gt_path = cloudnet_95_path / 'train_gt_additional_to38cloud'
    blue_dir = os.listdir(blue_path)

    nir_list = []
    blue_list = []
    green_list = []
    red_list = []
    gt_list = []

    image_list = []
    channel_list = []

    for blue_file in tqdm(blue_dir):
        base_name = '_'.join(blue_file.split('_')[1:])
        search_name = base_name.replace('.TIF', '').strip()
        if search_name in blue_file:
            image_list = []

            red_file = red_path / ('red_' + base_name)
            green_file = green_path / ('green_' + base_name)
            nir_file = nir_path / ('nir_' + base_name)
            blue_file = blue_path / ('blue_' + base_name)
            gt_file = gt_path / ('gt_' + base_name)

            image_list.append(red_file)
            image_list.append(green_file)
            image_list.append(nir_file)
            image_list.append(blue_file)
            channel_list.append(image_list)
            gt_list.append(gt_file)

    return channel_list, gt_list








if __name__ == "__main__":
    cloudnet_95_path = Path('../data/95-cloud_training_only_additional_to38-cloud')
    nonempty_cnet_95_path = cloudnet_95_path / 'training_patches_95-cloud_nonempty.csv'
    cloudnet_95_df = pd.read_csv(nonempty_cnet_95_path)

    cloudnet_38_path = Path('../data/38-Cloud_training')
    nonempty_cnet_38_path = cloudnet_38_path / 'training_patches_38-cloud_nonempty.csv'
    cloudnet_38_df = pd.read_csv(nonempty_cnet_38_path)

    cnet_38 = set(cloudnet_38_df['name'].to_list())
    cnet_95 = set(cloudnet_95_df['name'].to_list())
    additional = cnet_95 - cnet_38
    test_eval_df = pd.DataFrame(data={'name': list(additional)})
    test_eval_df.to_csv(cloudnet_95_path / 'evaluation_testing_patches_nonempty.csv')

    read_cloudnet(test_eval_df=test_eval_df, cloudnet_95_path=cloudnet_95_path)