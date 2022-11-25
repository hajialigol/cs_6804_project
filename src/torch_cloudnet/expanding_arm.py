import torch.nn as nn
from torch import add, concat, stack, sum


def add_block_exp_path(input_tensor1, input_tensor2, input_tensor3):
    x = stack([input_tensor1, input_tensor2, input_tensor3])
    return sum(x, dim=0)


class ConvBlock(nn.Module):
    """
    Maps to "conv_block_exp_path"
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, num_convs: int = 2):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_convs = num_convs
        self.conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding='same'
        )
        self.conv_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding='same'
        )
        self.relu = nn.ReLU(inplace=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.conv_batch_relu_1 = nn.Sequential(
            self.conv_1,
            self.batch_norm,
            self.relu
        )
        self.conv_batch_relu_1 = nn.Sequential(
            self.conv_2,
            self.batch_norm,
            self.relu
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        x = self.conv_2(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        if self.num_convs == 3:
            # do it for the third time
            x = self.conv_2(x)
            x = self.batch_norm(x)
            x = self.relu(x)
        return x


class ImproveFFBlock(nn.Module):
    """
    Maps to "improve_ff_block{x}", where x is either 1,2,3 or 4
    """

    def __init__(
            self,
            block1: bool = False,
            block2: bool = False,
            block3: bool = False,
            block4: bool = False
    ):
        super(ImproveFFBlock, self).__init__()
        self.block1 = block1
        self.block2 = block2
        self.block3 = block3
        self.block4 = block4
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(4, 4))
        self.max_pool_3 = nn.MaxPool2d(kernel_size=(8, 8))
        self.max_pool_4 = nn.MaxPool2d(kernel_size=(16, 16))
        self.relu = nn.ReLU(inplace=False)

    def skip_connection(self, layer, tensor, block_num):
        input_tensor = layer(tensor)
        x = layer(tensor)
        for i in range(block_num):
            x = concat((x, input_tensor), dim=1)
        return x

    def forward(
            self,
            input_tensor1,
            pure_ff,
            input_tensor2=None,
            input_tensor3=None,
            input_tensor4=None
    ):
        x1 = concat(
            (
                self.max_pool_1(input_tensor1),
                self.max_pool_1(input_tensor1),
            ),
            dim=1
        )

        # check if functionality should be with only one input
        if self.block1:
            x = add(x1, pure_ff)

        elif self.block2:
            x2 = self.skip_connection(layer=self.max_pool_2, tensor=input_tensor2, block_num=3)
            tensor_to_sum = stack([x1, x2, pure_ff])
            x = sum(tensor_to_sum, dim=0)

        elif self.block3:
            x2 = self.skip_connection(layer=self.max_pool_2, tensor=input_tensor2, block_num=3)
            x3 = self.skip_connection(layer=self.max_pool_3, tensor=input_tensor3, block_num=7)
            tensor_to_sum = stack([x1, x2, x3, pure_ff])
            x = sum(tensor_to_sum, dim=0)

        # block 4
        else:
            x2 = self.skip_connection(layer=self.max_pool_2, tensor=input_tensor2, block_num=3)
            x3 = self.skip_connection(layer=self.max_pool_3, tensor=input_tensor3, block_num=7)
            x4 = self.skip_connection(layer=self.max_pool_4, tensor=input_tensor4, block_num=15)
            tensor_to_sum = stack([x1, x2, x3, x4, pure_ff])
            x = sum(tensor_to_sum, dim=0)

        x = self.relu(x)
        return x
