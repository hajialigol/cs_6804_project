from argparse import ArgumentParser
from torch.cuda import is_available
from typing import Union
from cs_6804_project.src.torch_cloudnet.mid_fuse_conv_model import MFCCloudNet
from cs_6804_project.src.torch_cloudnet.mid_fuse_pool_model import MFPCloudNet
from cs_6804_project.src.torch_cloudnet.model import CloudNet


class TrainingArguments:
    def __init__(self, arg_parser: ArgumentParser):
        self.args = arg_parser.parse_args()
        self.batch_size = 4
        self.learning_rate = 0.001
        self.iterations = 3
        self.gpu_id = 0
        self.device = f'cuda:{self.gpu_id}' if is_available() else 'cpu'
        self.threshold_experiment = False
        self.threshold_validation_predictions = False
        self.args_dict = {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'iterations': self.iterations,
            'gpu_id': self.gpu_id,
            'device': self.device,
        }

    def parse_args(self):
        if 'batch_size' in self.args:
            self.batch_size = self.args.batch_size
            self.args_dict['batch_size'] = self.batch_size
        if 'learning_rate' in self.args:
            self.learning_rate = self.args.learning_rate
            self.args_dict['learning_rate'] = self.learning_rate
        if 'iterations' in self.args:
            self.iterations = self.args.iterations
            self.args_dict['iterations'] = self.iterations
        if 'gpu_id' in self.args:
            self.gpu_id = self.args.gpu_id
            self.args_dict['gpu_id'] = self.gpu_id

    def __getitem__(self, arg_key: str):
        return self.args_dict[arg_key]


class ThresholdArguments:
    def __init__(self, arg_parser: ArgumentParser):
        self.args = arg_parser.parse_args()
        self.best_model_path = ''
        self.best_model_type = ''
        self.gpu_id = 0
        self.device = f'cuda:{self.gpu_id}' if is_available() else 'cpu'
        self.threshold_experiment = False
        self.threshold_validation_predictions = False
        self.args_dict = {
            'best_model_path': self.best_model_path,
            'best_model_type': self.best_model_type,
            'threshold_experiment': self.threshold_experiment,
            'threshold_validation_predictions': self.threshold_validation_predictions,
            'gpu_id': self.gpu_id,
            'device': self.device
        }
        self.parse_args()

    def parse_args(self):
        if 'gpu_id' in self.args:
            self.gpu_id = self.args.gpu_id
            self.args_dict['gpu_id'] = self.gpu_id
        if 'best_model_path' in self.args:
            self.best_model_path = self.args.best_model_path
            self.args_dict['best_model_path'] = self.best_model_path
        if 'best_model_type' in self.args:
            self.best_model_type = self.args.best_model_type
            self.args_dict['best_model_type'] = self.best_model_type
        if 'threshold_experiment' in self.args:
            self.threshold_experiment = self.args.threshold_experiment
            self.args_dict['threshold_experiment'] = self.threshold_experiment
        if 'threshold_validation_predictions' in self.args:
            self.threshold_validation_predictions = self.args.threshold_validation_predictions
            self.args_dict['threshold_validation_predictions'] = self.threshold_validation_predictions
        if self.args_dict['threshold_validation_predictions'] and not self.args_dict['best_model_path']:
            raise ValueError('best_model_path needs to be given in order for validation predictions to be generated')
        if self.args_dict['threshold_validation_predictions'] and not self.args_dict['best_model_type']:
            raise ValueError('best_model_type needs to be given in order for validation predictions to be generated')

    def parse_best_model(self) -> Union[CloudNet, MFCCloudNet, MFCCloudNet]:
        """
        Method that returns the correct model based on the user's input
        :return: corresponding model
        :rtype: Union[CloudNet, MFCCloudNet, MFCCloudNet]
        """
        best_model = ''
        if 'early' in self.args_dict['best_model_type'].lower():
            best_model = CloudNet().to(self.device)
        if 'mid_conv' in self.args_dict['best_model_type'].lower():
            best_model = MFCCloudNet().to(self.device)
        if 'mid_pool' in self.args_dict['best_model_type'].lower():
            best_model = MFPCloudNet().to(self.device)
        else:
            ValueError('--best_model_type should be one of [early, mid_conv, mid_pool]')
        return best_model

    def __getitem__(self, arg_key: str):
        return self.args_dict[arg_key]

