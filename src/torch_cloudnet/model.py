from cs_6804_project.src.torch_cloudnet.expanding_arm import *
from cs_6804_project.src.torch_cloudnet.contracting_arm import *


class CloudNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.conv_init = nn.Conv2d(
            in_channels=4,
            out_channels=16,
            kernel_size=(3, 3),
            padding='same'
        )
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.contracting_1 = ContractingArm(
            in_channels=16,
            out_channels=32,
            kernel_size=3
        )
        self.contracting_2 = ContractingArm(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )
        self.contracting_3 = ContractingArm(
            in_channels=64,
            out_channels=128,
            kernel_size=3
        )
        self.contracting_4 = ContractingArm(
            in_channels=128,
            out_channels=256,
            kernel_size=3
        )
        self.improved_contracting = ImprovedContractingArm(
            in_channels=256,
            out_channels=512,
            kernel_size=3
        )
        self.bridge = Bridge(
            in_channels=512,
            out_channels=1024,
            kernel_size=3
        )
        self.conv2dt_512 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=(2, 2),
            # padding='same',
            stride=(2, 2)
        )
        self.conv2dt_256 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=(2, 2),
            # padding='same',
            stride=(2, 2)
        )
        self.conv2dt_128 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(2, 2),
            # padding='same',
            stride=(2, 2)
        )
        self.conv2dt_64 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=(2, 2),
            # padding='same',
            stride=(2, 2)
        )
        self.conv2dt_32 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=(2, 2),
            # padding='same',
            stride=(2, 2)
        )
        self.improve_ff_block_4 = ImproveFFBlock(block4=True)
        self.conv_block_7 = ConvBlock(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            num_convs=3
        )
        self.improve_ff_block_3 = ImproveFFBlock(block4=True)
        self.conv_block_8 = ConvBlock(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            num_convs=2
        )
        self.improve_ff_block_2 = ImproveFFBlock(block4=True)
        self.conv_block_9 = ConvBlock(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            num_convs=2
        )
        self.improve_ff_block_1 = ImproveFFBlock(block4=True)
        self.conv_block_10 = ConvBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            num_convs=2
        )
        self.conv_block_11 = ConvBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            num_convs=2
        )
        self.final_conv = nn.Conv2d(
            in_channels=32,
            out_channels=self.n_classes,
            kernel_size=(1, 1)
        )

    def forward(self, x):
        ## contracting arm
        conv_1 = self.contracting_1(x)
        conv_2 = self.contracting_1(conv_1)
        conv_3 = self.contracting_2(conv_2)
        conv_4 = self.contracting_3(conv_3)
        conv_5 = self.improved_contracting(conv_4)
        conv_6 = self.bridge(conv_5)

        ## expanding arm
        # 7th block
        conv_t7 = self.conv2dt_512(conv_6)
        prevup_7 = self.improve_ff_block_4(
            input_tensor1=conv_4,
            pure_ff=conv_5,
            input_tensor2=conv_3,
            input_tensor3=conv_2,
            input_tensor4=conv_1
        )
        up_7 = concat((conv_t7, prevup_7), dim=3)
        conv_7 = self.conv_block_7(up_7)
        conv_7 = self.relu(
            add_block_exp_path(input_tensor1=conv_7, input_tensor2=conv_5, input_tensor3=conv_t7)
        )

        # 8th block
        conv_t8 = self.conv2dt_256(conv_7)
        prevup_8 = self.improve_ff_block_3(
            input_tensor1=conv_3,
            pure_ff=conv_4,
            input_tensor2=conv_2,
            input_tensor3=conv_1
        )
        up_8 = concat((conv_t8, prevup_8), dim=3)
        conv_8 = self.conv_block_8(up_8)
        conv_8 = self.relu(
            add_block_exp_path(input_tensor1=conv_8, input_tensor2=conv_4, input_tensor3=conv_t8)
        )

        # 9th block
        conv_t9 = self.conv2dt_128(conv_8)
        prevup_9 = self.improve_ff_block_2(
            input_tensor1=conv_1,
            pure_ff=conv_3,
            input_tensor2=conv_2,
        )
        up_9 = concat((conv_t9, prevup_9), dim=3)
        conv_9 = self.conv_block_9(up_9)
        conv_9 = self.relu(
            add_block_exp_path(input_tensor1=conv_9, input_tensor2=conv_3, input_tensor3=conv_t9)
        )

        # 10th block
        conv_t10 = self.conv2dt_64(conv_9)
        prevup_10 = self.improve_ff_block_1(
            input_tensor1=conv_1,
            pure_ff=conv_2
        )
        up_10 = concat((conv_t10, prevup_10), dim=3)
        conv_10 = self.conv_block_10(up_10)
        conv_10 = self.relu(
            add_block_exp_path(input_tensor1=conv_10, input_tensor2=conv_2, input_tensor3=conv_t10)
        )

        # 11th block
        conv_t11 = self.conv2dt_32(conv_10)
        up_11 = concat((conv_t11, conv_1), dim=3)
        conv_11 = self.conv_block_11(up_11)
        conv_11 = self.relu(
            add_block_exp_path(input_tensor1=conv_11, input_tensor2=conv_1, input_tensor3=conv_t11)
        )

        # 12th block
        conv_12 = self.sigmoid(
            self.final_conv(conv_11)
        )
        return conv_12
