from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from cs_6804_project.src.torch_cloudnet.model import CloudNet
from cs_6804_project.src.torch_cloudnet.arguments import TrainingArguments


def train(model: CloudNet, data_loader: DataLoader, args: TrainingArguments):
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args['learning_rate'],
        momentum=0.9
    )
    model = model.to(args['device'])
    # gets rid of the 'Expected object of scalar type Double but got Float' error
    model = model.float()
    for data in tqdm(data_loader):
        batch_ims, labels = data
        batch_ims = batch_ims.to(args['device'])
        optimizer.zero_grad()
        outputs = model(batch_ims).to('cpu')
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
