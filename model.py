import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def conv_block(inp, oup, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0), pool_size=(1, 1, 1)):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size, stride, padding),
        nn.BatchNorm3d(oup),
        nn.ELU(),
        nn.AdaptiveMaxPool3d(pool_size),
        nn.Dropout3d(p=0.1),
    )


def sep_block(inp, oup, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0)):
    return nn.Sequential(
        nn.ConstantPad3d((1, 1, 2, 1, 1, 1), 0),
        nn.Conv3d(inp, inp, kernel_size, stride, padding, groups=inp),
        nn.Conv3d(inp, oup, (1, 1, 1), stride, padding),
        nn.BatchNorm3d(oup),
        nn.ELU(),
        nn.Dropout3d(p=0.1),
    )


def sep_block2(inp, oup, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0)):
    return nn.Sequential(
        nn.ConstantPad3d((1, 1, 2, 1, 1, 1), 0),
        nn.Conv3d(inp, inp, kernel_size, stride, padding, groups=inp),
        nn.Conv3d(inp, oup, (1, 1, 1), stride, padding),
        nn.BatchNorm3d(oup),
        nn.ELU(),
        nn.Dropout3d(p=0.1),
    )


def fc_layer(inp, oup):
    return nn.Sequential(
        nn.Linear(inp, oup, bias=True),
        nn.BatchNorm1d(oup),
        nn.ELU(),
        nn.Dropout(0.1)
    )


class CNN_with_CAMFB(nn.Module):
    def __init__(self):
        super(CNN_with_CAMFB, self).__init__()

        in_channels = 1

        self.conv1_left = conv_block(in_channels, 24, (11, 13, 11), 4, (6, 9, 6), (11, 27, 22))
        self.conv2_left = conv_block(24, 48, (5, 6, 5), 1, (2, 3, 2), (5, 13, 10))

        self.conv1_right = conv_block(in_channels, 24, (11, 13, 11), 4, (6, 9, 6), (11, 27, 22))
        self.conv2_right = conv_block(24, 48, (5, 6, 5), 1, (2, 3, 2), (5, 13, 10))

        self.conv3_sep_left = sep_block(48, 48, (3, 4, 3))
        self.conv4_sep_left = sep_block(48, 48, (3, 4, 3))
        self.conv5_sep_left = sep_block(48, 48, (3, 4, 3))

        self.conv3_sep_right = sep_block(48, 48, (3, 4, 3))
        self.conv4_sep_right = sep_block(48, 48, (3, 4, 3))
        self.conv5_sep_right = sep_block(48, 48, (3, 4, 3))

        self.conv6_sep = sep_block(96, 96, (3, 4, 3))
        self.conv7_sep = sep_block(96, 96, (3, 4, 3))
        self.conv8_sep = sep_block(96, 96, (3, 3, 3))

        self.conv6_mid = conv_block(96, 48, (3, 4, 3), 1, (1, 2, 1), (2, 6, 4))
        self.conv7_mid = conv_block(48, 24, (2, 4, 3), 1, (1, 1, 1), (1, 2, 1))
        self.conv6_sep = sep_block2(24, 24, (1, 1, 1))
        self.conv7_sep = sep_block2(24, 24, (1, 1, 1))
        self.conv8_sep = sep_block2(24, 24, (1, 1, 1))

        self.cli_data1 = fc_layer(11, 32)
        self.cli_data2 = fc_layer(32, 10)

        self.mri_fc = fc_layer(48, 10)
        self.mri_fc2 = fc_layer(96, 10)
        self.mri_fc3 = fc_layer(20, 10)

        self.proj_space = 50
        self.dataproj1 = nn.Linear(32, self.proj_space)
        self.dataproj2 = nn.Linear(32, self.proj_space)
        self.dataproj3 = nn.Linear(32, self.proj_space)
        self.dataproj4 = nn.Linear(32, self.proj_space)

        self.imgproj1 = nn.Linear(24, self.proj_space)
        self.imgproj2 = nn.Linear(24, self.proj_space)
        self.imgproj3 = nn.Linear(24, self.proj_space)
        self.imgproj4 = nn.Linear(24, self.proj_space)

        self.camfb_fc = fc_layer(20, 10)

        self.mfb_image = fc_layer(20, 10)
        self.mfb_data = fc_layer(20, 10)

        self.all_fc = fc_layer(30, 10)

        self.final_fc = nn.Linear(10, 1)

        self.initialize_weights()

    def forward(self, MRI_img, CLI_data):

        MRI_left = MRI_img[:, :, :91, :, :]
        MRI_right = MRI_img[:, :, 91:, :, :]

        MRI_right = self.conv1_right(MRI_right)
        MRI_right = self.conv2_right(MRI_right)

        MRI_left = self.conv1_left(MRI_left)
        MRI_left = self.conv2_left(MRI_left)

        MRI_left_sep = MRI_left
        MRI_right_sep = MRI_right

        MRI_left_sep = self.conv3_sep_left(MRI_left_sep)
        MRI_left_sep = self.conv4_sep_left(MRI_left_sep)
        MRI_left_sep = self.conv5_sep_left(MRI_left_sep)

        MRI_left = MRI_left + MRI_left_sep

        MRI_right_sep = self.conv3_sep_right(MRI_right_sep)
        MRI_right_sep = self.conv4_sep_right(MRI_right_sep)
        MRI_right_sep = self.conv5_sep_right(MRI_right_sep)

        MRI_right = MRI_right + MRI_right_sep

        MRI_mid = torch.cat([MRI_left, MRI_right], 1)

        MRI_mid = self.conv6_mid(MRI_mid)
        MRI_mid = self.conv7_mid(MRI_mid)

        # MRI_mid_sep = MRI_mid
        #
        # MRI_mid_sep = self.conv6_sep(MRI_mid_sep)
        # MRI_mid_sep = self.conv7_sep(MRI_mid_sep)
        # MRI_mid_sep = self.conv8_sep(MRI_mid_sep)
        #
        # MRI_mid = MRI_mid + MRI_mid_sep

        """
        # CAMFB 
        Separate channels for capturing more latent information
        """
        CAMFB_left_channel = MRI_mid[:, :12, :, :, :]
        CAMFB_right_channel = MRI_mid[:, 12:, :, :, :]

        CAMFB_left_channel = CAMFB_left_channel.view(CAMFB_left_channel.size(0), -1)
        CAMFB_right_channel = CAMFB_right_channel.view(CAMFB_right_channel.size(0), -1)

        CLI_data = self.cli_data1(CLI_data)
        cli = self.cli_data2(CLI_data)

        """
        # CAMFB left channels
        """
        data_left_data1 = self.dataproj1(CLI_data)
        data_right_data1 = self.dataproj2(CLI_data)
        CAMFB_left_MRI1 = self.imgproj1(CAMFB_left_channel)
        CAMFB_right_MRI1 = self.imgproj2(CAMFB_left_channel)

        # CAMFB layer
        Dropout_elt = nn.Dropout(p=0.1)

        size_x = CAMFB_left_MRI1.shape[1]
        size_z = data_left_data1.shape[1]

        camfb_left_eltwise1 = torch.mul(CAMFB_left_MRI1.view(-1, size_x, 1), data_left_data1.view(-1, 1, size_z))
        camfb_left_norm1 = F.normalize(camfb_left_eltwise1, p=2, dim=2)
        camfb_left_squeeze1 = torch.sum(camfb_left_norm1, 1)
        camfb_left_attention1 = camfb_left_squeeze1

        camfb_right_eltwise1 = torch.mul(CAMFB_right_MRI1, data_right_data1)
        camfb_mid_eltwise1 = torch.mul(camfb_right_eltwise1, camfb_left_attention1)
        camfb_mid_drop1 = Dropout_elt(camfb_mid_eltwise1)
        camfb_mid_resh1 = camfb_mid_drop1.view(-1, 10, 5)
        camfb_mid_squeeze1 = torch.sum(camfb_mid_resh1, 2)
        camfb_mid_out1 = torch.sqrt(F.relu(camfb_mid_squeeze1)) - torch.sqrt(F.relu(-camfb_mid_squeeze1))
        camfb_mid_out1 = F.normalize(camfb_mid_out1, p=2, dim=1)
        camfb_out_left = camfb_mid_out1

        """
        # CAMFB right channels
        """
        data_left_data2 = self.dataproj3(CLI_data)
        data_right_data2 = self.dataproj4(CLI_data)
        CAMFB_left_MRI2 = self.imgproj3(CAMFB_right_channel)
        CAMFB_right_MRI2 = self.imgproj4(CAMFB_right_channel)

        # CAMFB layer
        Dropout_elt = nn.Dropout(p=0.1)

        size_x = CAMFB_left_MRI2.shape[1]
        size_z = data_left_data2.shape[1]

        camfb_left_eltwise2 = torch.mul(CAMFB_left_MRI2.view(-1, size_x, 1), data_left_data2.view(-1, 1, size_z))
        camfb_left_norm2 = F.normalize(camfb_left_eltwise2, p=2, dim=2)

        camfb_left_squeeze2 = torch.sum(camfb_left_norm2, 1)

        camfb_left_attention2 = camfb_left_squeeze2

        camfb_right_eltwise2 = torch.mul(CAMFB_right_MRI2, data_right_data2)
        camfb_mid_eltwise2 = torch.mul(camfb_right_eltwise2, camfb_left_attention2)
        camfb_mid_drop2 = Dropout_elt(camfb_mid_eltwise2)
        camfb_mid_resh2 = camfb_mid_drop2.view(-1, 10, 5)
        camfb_mid_squeeze2 = torch.sum(camfb_mid_resh2, 2)
        camfb_mid_out2 = torch.sqrt(F.relu(camfb_mid_squeeze2)) - torch.sqrt(F.relu(-camfb_mid_squeeze2))
        camfb_mid_out2 = F.normalize(camfb_mid_out2, p=2, dim=1)
        camfb_out_right = camfb_mid_out2

        camfb_out = torch.cat([camfb_out_left, camfb_out_right], 1)

        camfb_out = self.camfb_fc(camfb_out)
        mri_out = torch.cat([CAMFB_left_channel, CAMFB_right_channel], 1)
        mri_out = self.mri_fc(mri_out)

        x = torch.cat([mri_out, cli, camfb_out], 1)

        x = self.all_fc(x)
        x = self.final_fc(x)

        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
