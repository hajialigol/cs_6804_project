from cs_6804_project.src.torch_cloudnet.expanding_arm import *
from cs_6804_project.src.torch_cloudnet.contracting_arm import *


class LFCloudNet(nn.Module):
    """
    Torch module responsible for representing the middle-fusion CloudNet architecture
    Adds pooling layers to fuse along contracting arm
    """
    def __init__(self, n_classes=1, share_sigmoid: bool=True):
        super().__init__()
        self.n_classes = n_classes
        self.share_sigmoid = share_sigmoid
        self.conv_init_rgb = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=(3, 3),
            padding='same'
        )
        self.conv_init_nir = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(3, 3),
            padding='same'
        )
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

        ### RGB Contracting Arm ###
        self.contracting_1_rgb = ContractingArm(
            in_channels=16,
            out_channels=32,
            kernel_size=3
        )
        self.contracting_2_rgb = ContractingArm(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )
        self.contracting_3_rgb = ContractingArm(
            in_channels=64,
            out_channels=128,
            kernel_size=3
        )
        self.contracting_4_rgb = ContractingArm(
            in_channels=128,
            out_channels=256,
            kernel_size=3
        )
        self.improved_contracting_rgb = ImprovedContractingArm(
            in_channels=256,
            out_channels=512,
            kernel_size=3
        )
        self.bridge_rgb = Bridge(
            in_channels=512,
            out_channels=1024,
            kernel_size=3
        )

        ### NIR Contracting Arm ###
        self.contracting_1_nir = ContractingArm(
            in_channels=16,
            out_channels=32,
            kernel_size=3
        )
        self.contracting_2_nir = ContractingArm(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )
        self.contracting_3_nir = ContractingArm(
            in_channels=64,
            out_channels=128,
            kernel_size=3
        )
        self.contracting_4_nir = ContractingArm(
            in_channels=128,
            out_channels=256,
            kernel_size=3
        )
        self.improved_contracting_nir = ImprovedContractingArm(
            in_channels=256,
            out_channels=512,
            kernel_size=3
        )
        self.bridge_nir = Bridge(
            in_channels=512,
            out_channels=1024,
            kernel_size=3
        )

        ### RGB Expanding Arm ###
        self.conv2dt_512_rgb = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        self.conv2dt_256_rgb = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        self.conv2dt_128_rgb = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        self.conv2dt_64_rgb = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        self.conv2dt_32_rgb = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        self.improve_ff_block_4_rgb = ImproveFFBlock(block4=True)
        self.conv_block_7_rgb = ConvBlock(
            in_channels=1024,
            out_channels=512,
            kernel_size=3,
            num_convs=3
        )
        self.improve_ff_block_3_rgb = ImproveFFBlock(block3=True)
        self.conv_block_8_rgb = ConvBlock(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            num_convs=2
        )
        self.improve_ff_block_2_rgb = ImproveFFBlock(block2=True)
        self.conv_block_9_rgb = ConvBlock(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            num_convs=2
        )
        self.improve_ff_block_1_rgb = ImproveFFBlock(block1=True)
        self.conv_block_10_rgb = ConvBlock(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            num_convs=2
        )
        self.conv_block_11_rgb = ConvBlock(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            num_convs=2
        )
        self.final_conv_rgb = nn.Conv2d(
                in_channels=32,
                out_channels=self.n_classes,
                kernel_size=(1, 1)
        )

        # NIR Expanding Arm
        self.conv2dt_512_nir = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        self.conv2dt_256_nir = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        self.conv2dt_128_nir = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        self.conv2dt_64_nir = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        self.conv2dt_32_nir = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        self.improve_ff_block_4_nir = ImproveFFBlock(block4=True)
        self.conv_block_7_nir = ConvBlock(
            in_channels=1024,
            out_channels=512,
            kernel_size=3,
            num_convs=3
        )
        self.improve_ff_block_3_nir = ImproveFFBlock(block3=True)
        self.conv_block_8_nir = ConvBlock(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            num_convs=2
        )
        self.improve_ff_block_2_nir = ImproveFFBlock(block2=True)
        self.conv_block_9_nir = ConvBlock(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            num_convs=2
        )
        self.improve_ff_block_1_nir = ImproveFFBlock(block1=True)
        self.conv_block_10_nir = ConvBlock(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            num_convs=2
        )
        self.conv_block_11_nir = ConvBlock(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            num_convs=2
        )
        self.final_conv_nir = nn.Conv2d(
                in_channels=32,
                out_channels=self.n_classes,
                kernel_size=(1, 1)
        )

    def forward(self, x):
        ## initial convolution
        conv_0_rgb = self.conv_init_rgb(x[:,:3].float())
        conv_0_nir = self.conv_init_nir(x[:,3:].float())

        ## contracting arm
        conv_1_rgb, pool_1_rgb = self.contracting_1_rgb(conv_0_rgb)
        conv_1_nir, pool_1_nir = self.contracting_1_nir(conv_0_nir)

        conv_2_rgb, pool_2_rgb = self.contracting_2_rgb(pool_1_rgb)
        conv_2_nir, pool_2_nir = self.contracting_2_nir(pool_1_nir)

        conv_3_rgb, pool_3_rgb = self.contracting_3_rgb(pool_2_rgb)
        conv_3_nir, pool_3_nir = self.contracting_3_nir(pool_2_nir)

        conv_4_rgb, pool_4_rgb = self.contracting_4_rgb(pool_3_rgb)
        conv_4_nir, pool_4_nir = self.contracting_4_nir(pool_3_nir)

        conv_5_rgb, pool_5_rgb = self.improved_contracting_rgb(pool_4_rgb)
        conv_5_nir, pool_5_nir = self.improved_contracting_nir(pool_4_nir)

        conv_6_rgb = self.bridge_rgb(pool_5_rgb)
        conv_6_nir = self.bridge_nir(pool_5_nir)

        ## expanding arm
        # 7th block rgb
        conv_t7_rgb = self.conv2dt_512_rgb(conv_6_rgb)
        prevup_7_rgb = self.improve_ff_block_4_rgb(
            input_tensor1=conv_4_rgb,
            pure_ff=conv_5_rgb,
            input_tensor2=conv_3_rgb,
            input_tensor3=conv_2_rgb,
            input_tensor4=conv_1_rgb
        )
        up_7_rgb = concat((conv_t7_rgb, prevup_7_rgb), dim=1)
        conv_7_rgb = self.conv_block_7_rgb(up_7_rgb)
        conv_7_rgb = self.relu(
            add_block_exp_path(input_tensor1=conv_7_rgb, input_tensor2=conv_5_rgb, input_tensor3=conv_t7_rgb)
        )
        # 7th block nir
        conv_t7_nir = self.conv2dt_512_nir(conv_6_nir)
        prevup_7_nir = self.improve_ff_block_4_nir(
            input_tensor1=conv_4_nir,
            pure_ff=conv_5_nir,
            input_tensor2=conv_3_nir,
            input_tensor3=conv_2_nir,
            input_tensor4=conv_1_nir
        )
        up_7_nir = concat((conv_t7_nir, prevup_7_nir), dim=1)
        conv_7_nir = self.conv_block_7_nir(up_7_nir)
        conv_7_nir = self.relu(
            add_block_exp_path(input_tensor1=conv_7_nir, input_tensor2=conv_5_nir, input_tensor3=conv_t7_nir)
        )

        # 8th block rgb
        conv_t8_rgb = self.conv2dt_256_rgb(conv_7_rgb)
        prevup_8_rgb = self.improve_ff_block_3_rgb(
            input_tensor1=conv_3_rgb,
            pure_ff=conv_4_rgb,
            input_tensor2=conv_2_rgb,
            input_tensor3=conv_1_rgb
            )
        up_8_rgb = concat((conv_t8_rgb, prevup_8_rgb), dim=1)
        conv_8_rgb = self.conv_block_8_rgb(up_8_rgb)
        conv_8_rgb = self.relu(
            add_block_exp_path(input_tensor1=conv_8_rgb, input_tensor2=conv_4_rgb, input_tensor3=conv_t8_rgb)
        )
        # 8th block nir
        conv_t8_nir = self.conv2dt_256_nir(conv_7_nir)
        prevup_8_nir = self.improve_ff_block_3_nir(
            input_tensor1=conv_3_nir,
            pure_ff=conv_4_nir,
            input_tensor2=conv_2_nir,
            input_tensor3=conv_1_nir
            )
        up_8_nir = concat((conv_t8_nir, prevup_8_nir), dim=1)
        conv_8_nir = self.conv_block_8_nir(up_8_nir)
        conv_8_nir = self.relu(
            add_block_exp_path(input_tensor1=conv_8_nir, input_tensor2=conv_4_nir, input_tensor3=conv_t8_nir)
        )

        # 9th block rgb
        conv_t9_rgb = self.conv2dt_128_rgb(conv_8_rgb)
        prevup_9_rgb = self.improve_ff_block_2_rgb(
            input_tensor1=conv_2_rgb,
            pure_ff=conv_3_rgb,
            input_tensor2=conv_1_rgb,
        )
        up_9_rgb = concat((conv_t9_rgb, prevup_9_rgb), dim=1)
        conv_9_rgb = self.conv_block_9_rgb(up_9_rgb)
        conv_9_rgb = self.relu(
            add_block_exp_path(input_tensor1=conv_9_rgb, input_tensor2=conv_3_rgb, input_tensor3=conv_t9_rgb)
        )
        # 9th block nir
        conv_t9_nir = self.conv2dt_128_nir(conv_8_nir)
        prevup_9_nir = self.improve_ff_block_2_nir(
            input_tensor1=conv_2_nir,
            pure_ff=conv_3_nir,
            input_tensor2=conv_1_nir,
        )
        up_9_nir = concat((conv_t9_nir, prevup_9_nir), dim=1)
        conv_9_nir = self.conv_block_9_nir(up_9_nir)
        conv_9_nir = self.relu(
            add_block_exp_path(input_tensor1=conv_9_nir, input_tensor2=conv_3_nir, input_tensor3=conv_t9_nir)
        )

        # 10th block rgb
        conv_t10_rgb = self.conv2dt_64_rgb(conv_9_rgb)
        prevup_10_rgb = self.improve_ff_block_1_rgb(
            input_tensor1=conv_1_rgb,
            pure_ff=conv_2_rgb
        )
        up_10_rgb = concat((conv_t10_rgb, prevup_10_rgb), dim=1)
        conv_10_rgb = self.conv_block_10_rgb(up_10_rgb)
        conv_10_rgb = self.relu(
            add_block_exp_path(input_tensor1=conv_10_rgb, input_tensor2=conv_2_rgb, input_tensor3=conv_t10_rgb)
        )
        # 10th block nir
        conv_t10_nir = self.conv2dt_64_nir(conv_9_nir)
        prevup_10_nir = self.improve_ff_block_1_nir(
            input_tensor1=conv_1_nir,
            pure_ff=conv_2_nir
        )
        up_10_nir = concat((conv_t10_nir, prevup_10_nir), dim=1)
        conv_10_nir = self.conv_block_10_nir(up_10_nir)
        conv_10_nir = self.relu(
            add_block_exp_path(input_tensor1=conv_10_nir, input_tensor2=conv_2_nir, input_tensor3=conv_t10_nir)
        )

        # 11th block rgb
        conv_t11_rgb = self.conv2dt_32_rgb(conv_10_rgb)
        up_11_rgb = concat((conv_t11_rgb, conv_1_rgb), dim=1)
        conv_11_rgb = self.conv_block_11_rgb(up_11_rgb)
        conv_11_rgb = self.relu(
            add_block_exp_path(input_tensor1=conv_11_rgb, input_tensor2=conv_1_rgb, input_tensor3=conv_t11_rgb)
        )
        # 11th block nir
        conv_t11_nir = self.conv2dt_32_nir(conv_10_nir)
        up_11_nir = concat((conv_t11_nir, conv_1_nir), dim=1)
        conv_11_nir = self.conv_block_11_nir(up_11_nir)
        conv_11_nir = self.relu(
            add_block_exp_path(input_tensor1=conv_11_nir, input_tensor2=conv_1_nir, input_tensor3=conv_t11_nir)
        )

        # 12th block
        if self.share_sigmoid:
            conv_12 = self.sigmoid(
                self.final_conv_rgb(conv_11_rgb) + self.final_conv_nir(conv_11_nir)
            )
        else:
            conv_12_rgb = self.sigmoid(
                self.final_conv_rgb(conv_11_rgb)
            )
            conv_12_nir = self.sigmoid(
                self.final_conv_nir(conv_11_nir)
            )
            conv_12 = (conv_12_rgb + conv_12_nir) / 2.0
        return conv_12
