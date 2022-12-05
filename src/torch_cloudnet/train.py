from datetime import datetime
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, \
    accuracy_score, recall_score
from torch import optim, save, tensor
from torch.utils.data import DataLoader
from torch.nn import Module
from torchmetrics.classification import BinaryJaccardIndex
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from cs_6804_project.src.torch_cloudnet.model import CloudNet
from cs_6804_project.src.torch_cloudnet.arguments import TrainingArguments
import numpy as np
import logging


def train(model: CloudNet, criterion: Module, writer: SummaryWriter,
          train_data: DataLoader, val_data: DataLoader, args, model_id=''):
    """
    High-level method responsible for training CloudNet model
    :param model: Model that the given data will be trained on
    :type model: CloudNet
    :param criterion: Loss function to be used in model training
    :type criterion: Module
    :param writer: Writer for the TensorBoard
    :type criterion: SummaryWriter
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
    optimizer = optim.SGD(
        model.parameters(),
        lr=args['learning_rate'],
        momentum=0.9
    )
    # gets rid of the 'Expected object of scalar type Double but got Float' error
    model = model.float()
    model.train()
    criterion = criterion.to(args['device'])
    running_loss = 0.0
    model_save_iters = 500
    record_loss_iters = 50
    iterations = args['iterations']
    i = 1
    for epoch in range(iterations):
        model = model.to(args['device'])
        writer.flush()
        for data in tqdm(train_data):
            batch_ims, labels, _ = data
            batch_ims = batch_ims.to(args['device'])
            optimizer.zero_grad()
            outputs = model(batch_ims).to('cpu')
            loss = criterion(y_true=labels, y_pred=outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % record_loss_iters == 0:
                running_loss /= record_loss_iters
                writer.add_scalar("Training Loss", running_loss, i // record_loss_iters)
                running_loss = 0
            if i % model_save_iters == 0:
                path_name = f'../../models/{model_id}cloudnet_epoch_{epoch}_iteration_{i}.pt'
                # create directory, if already existant
                Path('../../models').mkdir(parents=True, exist_ok=True)
                save(model.state_dict(), path_name)
                print(f'Saved model after {i} batches.')
            i += 1
        # Evaluate model after each epoch and log results
        eval_metric = eval(model, val_data)
        logging.debug(f'after epoch {epoch}, validation results are: {eval_metric}')
        print(f'average similarity for iteration {i} is {eval_metric}')
    # Save fully trained model
    path_name = f'../../models/{model_id}cloudnet_epoch_{iterations}_final.pt'
    # create directory, if already existant
    Path('../../models').mkdir(parents=True, exist_ok=True)
    save(model.state_dict(), path_name)
    print(f'Saved fully trained model.')

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
    jaccard = BinaryJaccardIndex()
    for data in tqdm(val_data):
        batch_ims, targets, _ = data
        outputs = model(batch_ims)
        results = outputs.detach().numpy().flatten()
        results = np.where(results > 0.05, 1, 0)
        labels = targets.detach().int().numpy().flatten()
        eval_metric = precision_recall_fscore_support(
            y_pred=results,
            y_true=labels
        )
        eval_dict['precision'] += eval_metric[0][1]
        eval_dict['recall'] += eval_metric[1][1]
        eval_dict['jaccard'] += jaccard(tensor(results), tensor(labels)).item()
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

def test(model, val_data, threshold):
    model.eval()
    model = model.to('cpu')
    eval_dict = {
        "jaccard": 0,
        "precision": 0,
        "recall": 0,
        "specificity": 0,
        "accuracy": 0
    }
    jaccard = BinaryJaccardIndex()
    for data in tqdm(val_data):
        batch_ims, targets, _ = data
        outputs = model(batch_ims)
        results = outputs.detach().numpy().flatten()
        results = np.where(results > threshold, 1, 0)
        labels = targets.detach().int().numpy().flatten()
        eval_metric = precision_recall_fscore_support(
            y_pred=results,
            y_true=labels
        )
        eval_dict['precision'] += eval_metric[0][1]
        eval_dict['recall'] += eval_metric[1][1]
        jacc_score = jaccard(tensor(results), tensor(labels)).item()
        if sum(results) == 0 and sum(labels) == 0:
            jacc_score = 1
        eval_dict['jaccard'] += jacc_score
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