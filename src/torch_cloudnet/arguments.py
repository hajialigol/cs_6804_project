from argparse import ArgumentParser
from torch.cuda import is_available


class TrainingArguments:
    def __init__(self, arg_parser: ArgumentParser):
        self.args = arg_parser.parse_args()
        self.batch_size = 4
        self.learning_rate = 0.001
        self.iterations = 3
        self.gpu_id = 0
        self.device = f'cuda:{self.gpu_id}' if is_available() else 'cpu'
        self.args_dict = {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'iterations': self.iterations,
            'gpu_id': self.gpu_id,
            'device': self.device
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
