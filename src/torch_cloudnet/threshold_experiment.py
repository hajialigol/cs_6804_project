from argparse import ArgumentParser
from datetime import datetime
from math import isnan
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import tensor, load
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryJaccardIndex
from typing import Dict, Union

from cs_6804_project.src.keras_cloudnet.utils import get_input_image_names
from cs_6804_project.src.torch_cloudnet.arguments import ThresholdArguments
from cs_6804_project.src.torch_cloudnet.dataset import CloudDataset
from cs_6804_project.src.torch_cloudnet.utils import save_predictions, read_predictions
from cs_6804_project.src.torch_cloudnet.model import CloudNet
from cs_6804_project.src.torch_cloudnet.mid_fuse_conv_model import MFCCloudNet
from cs_6804_project.src.torch_cloudnet.mid_fuse_pool_model import MFPCloudNet
from cs_6804_project.src.torch_cloudnet.late_fuse_model import LFCloudNet

import numpy as np
import pandas as pd
import json


def make_predictions(
        best_model: Union[CloudNet, MFPCloudNet, MFCCloudNet, LFCloudNet],
        args, #: ThresholdArguments,
        validation_loader: DataLoader,
        predictions_fname: str = 'best_predictions.pickle',
):
    """
    Method for making the predictions that will be fed into the
    :param best_model: The (best) model that will be used for the
                       threshold experiment
    :type best_model: Union[CloudNet, MFPCloudNet, MFCCloudNet]
    :param validation_loader: DataLoader holding the validation set
    :type validation_loader: DataLoader
    :param predictions_fname: Name of file to output predictions to
    :type predictions_fname: str
    :return:
    """
    best_model.eval()
    predictions_dict = dict()
    Path('../../data/best_validation_predictions').mkdir(parents=True, exist_ok=True)
    predictions_path = Path(f'../../data/best_validation_predictions/{predictions_fname}')
    for data in tqdm(validation_loader):
        batch_ims, labels, batch_fnames = data
        batch_ims = batch_ims.to(args['device'])
        outputs = best_model(batch_ims)
        # handles situation if the batch isn't evenly divisible by the amount of records
        predictions_batch_size = data[0].shape[0]
        for j in range(predictions_batch_size):
            # returns a list of a tuple... yeah, idk lmao
            fname = batch_fnames[0][j]
            image_dict = {
                "output": outputs[j].detach(),
                "labels": labels[j].detach()
            }
            predictions_dict[fname] = image_dict
    save_predictions(predictions_dict=predictions_dict, write_path=predictions_path)


def optimize_threshold(predictions_fname: str = 'best_predictions.pickle') -> Dict:
    """
    Method that handles the threshold optimization experiment
    :return: Dictionary containing threshold experiment results
    :type: Dict[float]
    """
    predictions_path = Path(f'../../data/best_validation_predictions/{predictions_fname}')
    predictions = read_predictions(read_path=predictions_path)
    threshold_dict = {round(0.001 + i/1000, 3): 0 for i in range(999)}
    jaccard = BinaryJaccardIndex()
    threshold = 0.001
    i = 0
    while threshold <= 0.999:
        for fname, vals in tqdm(predictions.items()):
            outputs = vals['output'].cpu()
            labels = vals['labels'].cpu()
            labels = labels.int().numpy().flatten()
            results = outputs.numpy().flatten()
            results = np.where(results > threshold, 1, 0)
            score = jaccard(tensor(results), tensor(labels)).item()
            if isnan(score):
                if sum(results) == 0 and sum(labels) == 0:
                    score = 1
                else:
                    print(f'score is {score}, sum of results and labels are {sum(results)} and {sum(labels)}')
            threshold_jaccard = score
            threshold_dict[threshold] += threshold_jaccard
        print(f'threshold: {threshold}, score sum: {threshold_dict[threshold]}')
        threshold = round(threshold + 0.001, 3)
        i += 1
        # pprint(f"results: {threshold_dict}")
    threshold_dict = {k: v/len(predictions) for k,v in threshold_dict.items()}
    return threshold_dict


if __name__ == "__main__":
    today = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    parser = ArgumentParser()
    parser.add_argument('--best_model_path', type=str, required=False)
    parser.add_argument('--best_model_type', type=str, required=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--threshold_experiment', type=bool, default=False, required=False)
    parser.add_argument('--threshold_validation_predictions', type=bool, default=False, required=False)
    args = ThresholdArguments(arg_parser=parser)
    full_model_path = Path('../../models') / args['best_model_path']
    best_model = args.parse_best_model()
    best_model.load_state_dict(load(full_model_path))

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

    # getting input images names
    train_patches_csv_name = 'training_patches_38-cloud_nonempty.csv'
    df_train_img = pd.read_csv(TRAIN_FOLDER / train_patches_csv_name)

    train_img, train_msk = get_input_image_names(df_train_img, TRAIN_FOLDER, if_train=True)
    # train_img = train_img[:100]
    # train_msk = train_msk[:100]

    # Split data into training and validation
    train_img_split, val_img_split, train_msk_split, val_msk_split = train_test_split(
        train_img, train_msk, test_size=val_ratio, random_state=42, shuffle=True
    )

    # Get datasets from file names
    ds_train = CloudDataset(train_img_split, train_msk_split, in_rows, in_cols, max_bit, transform=True)
    ds_val = CloudDataset(val_img_split, val_msk_split, in_rows, in_cols, max_bit)

    train_dataloader = DataLoader(ds_train, batch_size=4, shuffle=False)
    val_dataloader = DataLoader(ds_val, batch_size=4, shuffle=False)

    if args['threshold_validation_predictions']:
        make_predictions(best_model=best_model, validation_loader=val_dataloader, args=args)
    if args['threshold_experiment']:
        threshold_path = '../../threshold_experiment'
        Path(threshold_path).mkdir(parents=True, exist_ok=True)
        threshold_results = optimize_threshold()
        with open(Path(f'{threshold_path}/results_{today}.json'), 'w') as fp:
            json.dump(threshold_results, fp, indent=4)
