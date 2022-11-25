from datetime import datetime
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support,\
    jaccard_score, accuracy_score, recall_score
from torch import optim, save
from torch.utils.data import DataLoader
from tqdm import tqdm
from cs_6804_project.src.torch_cloudnet.model import CloudNet
from cs_6804_project.src.torch_cloudnet.arguments import TrainingArguments
from cs_6804_project.src.keras_cloudnet.losses import jacc_coef_pt
import numpy as np
import logging


def train(model: CloudNet, train_data: DataLoader, val_data: DataLoader,
          args: TrainingArguments):
    """
    High-level method responsible for training CloudNet model
    :param model: Model that the given data will be trained on
    :type model: CloudNet
    :param data_loader: Dataset used in training
    :type data_loader: DataLoader
    :param args: Arguments specified by the user for training
    :type args: TrainingArguments
    :return:
    """
    today = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    Path('../../logs').mkdir(parents=True, exist_ok=True)
    cur_log_file = Path('../../logs') / Path(f'train_log_{today}.log')
    logging.basicConfig(
        filename=cur_log_file,
        filemode='w',
        level=logging.DEBUG
    )
    running_loss = 0
    optimizer = optim.SGD(
        model.parameters(),
        lr=args['learning_rate'],
        momentum=0.9
    )
    # gets rid of the 'Expected object of scalar type Double but got Float' error
    model = model.float()
    model.train()
    i = 0
    for epoch in range(args['iterations']):
        model = model.to(args['device'])
        for data in tqdm(train_data):
            batch_ims, labels = data
            batch_ims = batch_ims.to(args['device'])
            optimizer.zero_grad()
            outputs = model(batch_ims).to('cpu')
            loss = jacc_coef_pt(y_true=labels, y_pred=outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 500 == 0:  # print every 2000 mini-batches
                print(f'iteration {i} loss: {running_loss / 50:.3f}')
                running_loss = 0.0
                path_name = f'../../models/cloudnet_epoch_{epoch}_iteration_{i}.pt'
                # create directory, if already existant
                Path('../../models').mkdir(parents=True, exist_ok=True)
                save(model.state_dict(), path_name)
            i += 1
        eval_metric = eval(model, val_data)
        logging.debug(f'after epoch {epoch}, validation results are: {eval_metric}')
        print(f'average similarity for iteration {i} is {eval_metric}')


def eval(model: CloudNet, val_data: DataLoader):
    model.eval()
    model = model.to('cpu')
    eval_dict = {
        "jaccard": 0,
        "precision": 0,
        "recall": 0,
        "specificity": 0,
        "accuracy": 0
    }
    for data in tqdm(val_data):
        batch_ims, labels = data
        results = model(batch_ims).detach().numpy().flatten()
        results = np.where(results > 0.5, 1, 0)
        labels = labels.detach().numpy().flatten()
        eval_metric = precision_recall_fscore_support(
            y_pred=results,
            y_true=labels
        )
        eval_dict['precision'] += eval_metric[0][1]
        eval_dict['recall'] += eval_metric[1][1]
        eval_dict['specificity'] += jaccard_score(
            y_pred=results,
            y_true=labels
        )
        eval_dict['accuracy'] += accuracy_score(
            y_pred=results,
            y_true=labels
        )
        eval_dict['specificity'] += recall_score(
            y_true=results,
            y_pred=labels,
            pos_label=0
        )
    # take average performance
    eval_dict = {k: v/len(val_data) for k,v in eval_dict.items()}
    return eval_dict
