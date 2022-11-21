import torch.nn as nn
from torch import add, concat, stack, sum


class ContractingArm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ContractingArm, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        """
        for first forward pass, you'll have an input_tensor of size (192, 192, 16). you'll have to
        explicitly state 
        - in_channels = 16
        - out_channels = 32 (same as filters in tf)
        - kernel_size = 3 (same as kernel_size in tf)
        """
        self.red_conv_1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding="same"
        )
        self.red_conv_2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,  # yes, this is on purpose
            kernel_size=(self.kernel_size, self.kernel_size),
            padding="same"
        )
        self.green_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=(1, 1),
            padding='same'
        )
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.batch_relu = nn.Sequential(
            self.batch_norm,
            self.relu
        )

    def forward(self, x):
        red_1 = self.red_conv_1(x)
        red_1 = self.batch_relu(red_1)
        red_2 = self.red_conv_2(red_1)
        red_2 = self.batch_relu(red_2)
        green = self.green_conv(x)
        green = self.batch_relu(green)
        # will have to check out the dimension
        orange = concat((x, green), dim=3)
        # think this is right
        blue = add(red_2, orange)
        blue = self.relu(blue)
        purple = self.max_pool(blue)
        return purple


class Bridge(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(Bridge, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.red_conv_1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding="same"
        )
        self.red_conv_2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,  # yes, this is on purpose
            kernel_size=(self.kernel_size, self.kernel_size),
            padding="same"
        )
        self.dropout = nn.Dropout(p=0.15)
        self.green_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=(1, 1),
            padding='same'
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.batch_relu = nn.Sequential(
            self.batch_norm,
            self.relu
        )

    def forward(self, x):
        red_1 = self.red_conv_1(x)
        red_1 = self.batch_relu(red_1)
        red_2 = self.red_conv_2(red_1)
        red_2 = self.batch_relu(red_2)
        green = self.green_conv(x)
        green = self.batch_relu(green)
        # will have to check out the dimension
        orange = concat((x, green), dim=3)
        # think this is right
        blue = add(red_2, orange)
        blue = self.relu(blue)
        return blue


class ImprovedContractingArm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ImprovedContractingArm, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.red_conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding="same"
        )
        self.red_conv_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding="same"
        )
        # same thing as red_conv_2, but implemented to make
        # it look more obvious
        self.red_conv_3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding="same"
        )
        self.top_green_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1),
            padding="same"
        )
        # same thing as top_green_conv,
        # but again, to make things look simpler
        self.bottom_green_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1),
            padding="same"
        )
        self.relu = nn.ReLU(inplace=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.batch_relu = nn.Sequential(
            self.batch_norm,
            self.relu
        )
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        red_1 = self.red_conv_1(x)
        red_1 = self.batch_relu(red_1)
        red_2 = self.red_conv_2(red_1)
        red_2 = self.red_conv_2(red_2)
        red_3 = self.red_conv_3(red_2)
        red_3 = self.red_conv_3(red_3)
        green_top = self.top_green_conv(x)
        green_top = self.batch_relu(green_top)
        orange = concat((x, green_top), dim=3)
        green_bottom = self.bottom_green_conv(red_2)
        green_bottom = self.batch_relu(green_bottom)
        tensor_to_add = stack([red_3, orange, green_bottom])
        blue = sum(tensor_to_add, dim=0)
        blue = self.relu(blue)
        purple = self.max_pool(blue)
        return purple
