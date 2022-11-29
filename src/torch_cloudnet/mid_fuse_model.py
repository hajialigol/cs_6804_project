from cs_6804_project.src.torch_cloudnet.expanding_arm import *
from cs_6804_project.src.torch_cloudnet.contracting_arm import *


class MFCloudNet(nn.Module):
    """
    Torch module responsible for representing the middle-fusion CloudNet architecture
    """
    def __init__(self, n_classes=1):
        super().__init__()
        self.n_classes = n_classes
        self.conv_init_rgb = nn.Conv2d(
            in_channels=4,
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

        ### RBG Contracting Arm ###
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

        ### Mutual Expanding Arm ###
        self.conv2dt_512 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        self.conv2dt_256 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        self.conv2dt_128 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        self.conv2dt_64 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        self.conv2dt_32 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        self.improve_ff_block_4 = ImproveFFBlock(block4=True)
        self.conv_block_7 = ConvBlock(
            in_channels=1024,
            out_channels=512,
            kernel_size=3,
            num_convs=3
        )
        self.improve_ff_block_3 = ImproveFFBlock(block3=True)
        self.conv_block_8 = ConvBlock(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            num_convs=2
        )
        self.improve_ff_block_2 = ImproveFFBlock(block2=True)
        self.conv_block_9 = ConvBlock(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            num_convs=2
        )
        self.improve_ff_block_1 = ImproveFFBlock(block1=True)
        self.conv_block_10 = ConvBlock(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            num_convs=2
        )
        self.conv_block_11 = ConvBlock(
            in_channels=64,
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
        ## initial convolution
        conv_0_rgb = self.conv_init_rgb(x[:,:3].float())
        conv_0_nir = self.conv_init_nir(x[:,3].float())
        conv_0_rgb = conv_0_rgb + conv_0_nir

        ## contracting arm
        conv_1_rgb, pool_1_rgb = self.contracting_1_rgb(conv_0_rgb)
        conv_1_nir, pool_1_nir = self.contracting_1_nir(conv_0_nir)
        conv_1_rgb = conv_1_rgb + conv_1_nir

        conv_2_rgb, pool_2_rgb = self.contracting_2_rgb(pool_1_rgb)
        conv_2_nir, pool_2_nir = self.contracting_1_nir(pool_1_nir)
        conv_2_rgb = conv_2_rgb + conv_2_nir
        pool_2_rgb = pool_2_rgb + pool_2_nir

        conv_3_rgb, pool_3_rgb = self.contracting_3_rgb(pool_2_rgb)
        conv_3_nir, pool_3_nir = self.contracting_3_nir(pool_2_nir)
        conv_3_rgb = conv_3_rgb + conv_3_nir
        pool_3_rgb = pool_3_rgb + pool_3_nir

        conv_4_rgb, pool_4_rgb = self.contracting_4_rgb(pool_3_rgb)
        conv_4_nir, pool_4_nir = self.contracting_4_nir(pool_3_nir)
        conv_4_rgb = conv_4_rgb + conv_4_nir
        pool_4_rgb = pool_4_rgb + pool_4_nir

        conv_5_rgb, pool_5_rgb = self.improved_contracting_rgb(pool_4_rgb)
        conv_5_nir, pool_5_nir = self.improved_contracting_nir(pool_4_nir)
        conv_5_rgb = conv_5_rgb + conv_5_nir
        pool_5_rgb = pool_5_rgb + pool_5_nir

        conv_6_rgb = self.bridge_rgb(pool_5_rgb)
        conv_6_nir = self.bridge_nir(pool_5_nir)
        conv_6_rgb = conv_6_rgb + conv_6_nir

        ## expanding arm
        # 7th block
        conv_t7 = self.conv2dt_512(conv_6_rgb)
        prevup_7 = self.improve_ff_block_4(
            input_tensor1=conv_4_rgb,
            pure_ff=conv_5_rgb,
            input_tensor2=conv_3_rgb,
            input_tensor3=conv_2_rgb,
            input_tensor4=conv_1_rgb
        )
        up_7 = concat((conv_t7, prevup_7), dim=1)
        conv_7 = self.conv_block_7(up_7)
        conv_7 = self.relu(
            add_block_exp_path(input_tensor1=conv_7, input_tensor2=conv_5_rgb, input_tensor3=conv_t7)
        )

        # 8th block
        conv_t8 = self.conv2dt_256(conv_7)
        prevup_8 = self.improve_ff_block_3(
            input_tensor1=conv_3_rgb,
            pure_ff=conv_4_rgb,
            input_tensor2=conv_2_rgb,
            input_tensor3=conv_1_rgb
            )
        up_8 = concat((conv_t8, prevup_8), dim=1)
        conv_8 = self.conv_block_8(up_8)
        conv_8 = self.relu(
            add_block_exp_path(input_tensor1=conv_8, input_tensor2=conv_4_rgb, input_tensor3=conv_t8)
        )

        # 9th block
        conv_t9 = self.conv2dt_128(conv_8)
        prevup_9 = self.improve_ff_block_2(
            input_tensor1=conv_2_rgb,
            pure_ff=conv_3_rgb,
            input_tensor2=conv_1_rgb,
        )
        up_9 = concat((conv_t9, prevup_9), dim=1)
        conv_9 = self.conv_block_9(up_9)
        conv_9 = self.relu(
            add_block_exp_path(input_tensor1=conv_9, input_tensor2=conv_3_rgb, input_tensor3=conv_t9)
        )

        # 10th block
        conv_t10 = self.conv2dt_64(conv_9)
        prevup_10 = self.improve_ff_block_1(
            input_tensor1=conv_1_rgb,
            pure_ff=conv_2_rgb
        )
        up_10 = concat((conv_t10, prevup_10), dim=1)
        conv_10 = self.conv_block_10(up_10)
        conv_10 = self.relu(
            add_block_exp_path(input_tensor1=conv_10, input_tensor2=conv_2_rgb, input_tensor3=conv_t10)
        )

        # 11th block
        conv_t11 = self.conv2dt_32(conv_10)
        up_11 = concat((conv_t11, conv_1_rgb), dim=1)
        conv_11 = self.conv_block_11(up_11)
        conv_11 = self.relu(
            add_block_exp_path(input_tensor1=conv_11, input_tensor2=conv_1_rgb, input_tensor3=conv_t11)
        )

        # 12th block
        conv_12 = self.sigmoid(
            self.final_conv(conv_11)
        )
        return conv_12
