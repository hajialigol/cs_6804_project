import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from cs_6804_project.src.keras_cloudnet.utils import get_input_image_names
from cs_6804_project.src.torch_cloudnet.dataset import CloudDataset
from cs_6804_project.src.torch_cloudnet.model import CloudNet
from cs_6804_project.src.torch_cloudnet.arguments import TrainingArguments
from cs_6804_project.src.torch_cloudnet.train import train
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--iterations', type=int, default=3)
    # just in case there are multiple gpus to your disposal
    # if gpu_id = 0, then you just use the only/first gpu you have
    parser.add_argument('--gpu_id', type=int, default=0)
    training_args = TrainingArguments(arg_parser=parser)
    GLOBAL_PATH = Path('../../data')
    TRAIN_FOLDER = GLOBAL_PATH / '38-Cloud_training'
    TEST_FOLDER = GLOBAL_PATH / '38-Cloud_test'
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
    experiment_name = "Cloud-Net"
    weights_partial = experiment_name + '.h5'
    weights_path = GLOBAL_PATH / weights_partial
    # weights_path = os.path.join(GLOBAL_PATH, experiment_name + '.h5')
    train_resume = False

    # getting input images names
    train_patches_csv_name = 'training_patches_38-cloud_nonempty.csv'
    df_train_img = pd.read_csv(TRAIN_FOLDER / train_patches_csv_name)
    # df_train_img = pd.read_csv(os.path.join(TRAIN_FOLDER, train_patches_csv_name))

    train_img, train_msk = get_input_image_names(df_train_img, TRAIN_FOLDER, if_train=True)

    # Split data into training and validation
    train_img_split, val_img_split, train_msk_split, val_msk_split = train_test_split(
        train_img, train_msk, test_size=val_ratio, random_state=42, shuffle=True
    )

    # Get datasets from file names
    ds_train = CloudDataset(train_img_split, train_msk_split, in_rows, in_cols, max_bit, transform=True)
    ds_val = CloudDataset(val_img_split, val_msk_split, in_rows, in_cols, max_bit)

    train_dataloader = DataLoader(ds_train, batch_size=4, shuffle=False)
    test_dataloader = DataLoader(ds_val, batch_size=4, shuffle=False)
    model = CloudNet()
    train(model=model, data_loader=train_dataloader, args=training_args)
