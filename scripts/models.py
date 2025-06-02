import torch
import torch.nn as nn
from layers import DoubleConvMaSA, NexBlock
import logging

logger = logging.getLogger(__name__)

class AutoencoderMaSA(nn.Module):
    def __init__(self, height, width, channels, savename="autoencoder_masa", use_masa=True, gamma=0.9):
        super(AutoencoderMaSA, self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.savename = savename
        filters = 32  # Keep reduced filters
        dropout_val = 0

        # Encoder
        self.conv_224 = DoubleConvMaSA(channels, filters)
        self.conv_224_att = DoubleConvMaSA(channels, filters, use_masa=use_masa, gamma=gamma)
        self.pool_112 = nn.MaxPool2d(2)
        self.conv_112 = DoubleConvMaSA(filters, 2 * filters)
        self.conv_112_att = DoubleConvMaSA(filters, 2 * filters, use_masa=use_masa, gamma=gamma)
        self.pool_56 = nn.MaxPool2d(2)
        self.conv_56 = DoubleConvMaSA(2 * filters, 4 * filters)
        self.conv_56_att = DoubleConvMaSA(2 * filters, 4 * filters, use_masa=use_masa, gamma=gamma)
        self.pool_28 = nn.MaxPool2d(2)
        self.conv_28 = DoubleConvMaSA(4 * filters, 8 * filters)
        self.pool_14 = nn.MaxPool2d(2)
        self.conv_14 = DoubleConvMaSA(8 * filters, 16 * filters)
        self.pool_7 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(16 * filters, 32 * filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(32 * filters),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.up_14 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_14 = DoubleConvMaSA(32 * filters + 16 * filters, 16 * filters)
        self.up_28 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_28 = DoubleConvMaSA(16 * filters + 8 * filters, 8 * filters)
        self.up_56 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_56 = DoubleConvMaSA(8 * filters + 4 * filters, 4 * filters)
        self.up_112 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_112 = DoubleConvMaSA(4 * filters + 2 * filters, 2 * filters)
        self.up_224 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_224 = DoubleConvMaSA(2 * filters + filters, filters, dropout=dropout_val)

        # Final layer
        self.conv_final = nn.Conv2d(filters, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        conv_224 = self.conv_224(x)
        conv_224_att = self.conv_224_att(x)
        pool_112 = self.pool_112(conv_224)
        conv_112 = self.conv_112(pool_112)
        conv_112_att = self.conv_112_att(pool_112)
        pool_56 = self.pool_56(conv_112)
        conv_56 = self.conv_56(pool_56)
        conv_56_att = self.conv_56_att(pool_56)
        pool_28 = self.pool_28(conv_56)
        conv_28 = self.conv_28(pool_28)
        pool_14 = self.pool_14(conv_28)
        conv_14 = self.conv_14(pool_14)
        pool_7 = self.pool_7(conv_14)

        # Bottleneck
        conv_7 = self.bottleneck(pool_7)

        # Decoder with skip connections
        up_14 = torch.cat([self.up_14(conv_7), conv_14], dim=1)
        up_conv_14 = self.up_conv_14(up_14)
        up_28 = torch.cat([self.up_28(up_conv_14), conv_28], dim=1)
        up_conv_28 = self.up_conv_28(up_28)
        up_56 = torch.cat([self.up_56(up_conv_28), conv_56_att], dim=1)
        up_conv_56 = self.up_conv_56(up_56)
        up_112 = torch.cat([self.up_112(up_conv_56), conv_112_att], dim=1)
        up_conv_112 = self.up_conv_112(up_112)
        up_224 = torch.cat([self.up_224(up_conv_112), conv_224_att], dim=1)
        up_conv_224 = self.up_conv_224(up_224)

        # Reconstruction
        conv_final = self.conv_final(up_conv_224)
        output = self.sigmoid(conv_final)
        return output, conv_7

    def get_embedding(self, x):
        conv_224 = self.conv_224(x)
        pool_112 = self.pool_112(conv_224)
        conv_112 = self.conv_112(pool_112)
        pool_56 = self.pool_56(conv_112)
        conv_56 = self.conv_56(pool_56)
        pool_28 = self.pool_28(conv_56)
        conv_28 = self.conv_28(pool_28)
        pool_14 = self.pool_14(conv_28)
        conv_14 = self.conv_14(pool_14)
        pool_7 = self.pool_7(conv_14)
        embedding = self.bottleneck(pool_7)
        return embedding

class NexNet_Seg(nn.Module):
    def __init__(self, input_shape, filters, kernel_sizes, depth, num_classes, dataset, dr=0, use_masa=True, gamma=0.9):
        super(NexNet_Seg, self).__init__()
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.depth = depth
        self.num_classes = num_classes
        self.dataset = dataset
        self.dr = dr
        self.use_masa = use_masa
        self.gamma = gamma

        # Encoder
        self.nb_1 = NexBlock(input_shape[2], filters[0], kernel_sizes[0], dilation_rate=2, use_masa=use_masa, gamma=gamma)
        self.pool_112 = nn.MaxPool2d(2)
        self.nb_2 = NexBlock(filters[0], filters[1], kernel_sizes[1], dilation_rate=2, use_masa=use_masa, gamma=gamma)
        self.pool_56 = nn.MaxPool2d(2)
        self.nb_3 = NexBlock(filters[1], filters[2], kernel_sizes[2], dilation_rate=2, use_masa=use_masa, gamma=gamma)
        self.pool_28 = nn.MaxPool2d(2)
        self.nb_4 = NexBlock(filters[2], filters[3], kernel_sizes[3], dilation_rate=2, use_masa=use_masa, gamma=gamma)
        self.pool_14 = nn.MaxPool2d(2)
        self.nb_5 = NexBlock(filters[3], filters[4], kernel_sizes[4], dilation_rate=2, use_masa=use_masa, gamma=gamma)
        self.pool_7 = nn.MaxPool2d(2)
        self.nb_6 = NexBlock(filters[4], filters[5], kernel_sizes[5], dilation_rate=2, use_masa=use_masa, gamma=gamma)

        # Decoder
        self.up_14 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_14 = DoubleConvMaSA(filters[5] + filters[4], filters[4])
        self.up_28 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_28 = DoubleConvMaSA(filters[4] + filters[3], filters[3])
        self.up_56 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_56 = DoubleConvMaSA(filters[3] + filters[2], filters[2])
        self.up_112 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_112 = DoubleConvMaSA(filters[2] + filters[1], filters[1])
        self.up_224 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_224 = DoubleConvMaSA(filters[1] + filters[0], filters[0], dropout=dr)

        # Final layer (output raw logits)
        self.conv_final = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        nb_1, nb_1_att = self.nb_1(x)
        pool_112 = self.pool_112(nb_1)
        nb_2, nb_2_att = self.nb_2(pool_112)
        pool_56 = self.pool_56(nb_2)
        nb_3, nb_3_att = self.nb_3(pool_56)
        pool_28 = self.pool_28(nb_3)
        nb_4, nb_4_att = self.nb_4(pool_28)
        pool_14 = self.pool_14(nb_4)
        nb_5, nb_5_att = self.nb_5(pool_14)
        pool_7 = self.pool_7(nb_5)
        nb_6, nb_6_att = self.nb_6(pool_7)

        # Decoder with skip connections
        up_14 = torch.cat([self.up_14(nb_6), nb_5_att], dim=1)
        up_conv_14 = self.up_conv_14(up_14)
        up_28 = torch.cat([self.up_28(up_conv_14), nb_4_att], dim=1)
        up_conv_28 = self.up_conv_28(up_28)
        up_56 = torch.cat([self.up_56(up_conv_28), nb_3_att], dim=1)
        up_conv_56 = self.up_conv_56(up_56)
        up_112 = torch.cat([self.up_112(up_conv_56), nb_2_att], dim=1)
        up_conv_112 = self.up_conv_112(up_112)
        up_224 = torch.cat([self.up_224(up_conv_112), nb_1_att], dim=1)
        up_conv_224 = self.up_conv_224(up_224)

        # Output raw logits
        output = self.conv_final(up_conv_224)
        return output

class NexNet_Seg_ssl(nn.Module):
    def __init__(self, input_shape, filters, kernel_sizes, depth, num_classes, dataset, encoder, dr=0, use_masa=True, gamma=0.9, freeze_encoder=True):
        super(NexNet_Seg_ssl, self).__init__()
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.depth = depth
        self.num_classes = num_classes
        self.dataset = dataset
        self.encoder = encoder
        self.dr = dr
        self.use_masa = use_masa
        self.gamma = gamma

        # Define the number of channels in the encoder's bottleneck output
        self.encoder_bottleneck_channels = 32 * 32  # From AutoencoderMaSA: filters=32, so 32 * filters = 1024

        # Conditionally freeze the encoder weights
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder weights frozen in NexNet_Seg_ssl.")
        else:
            logger.info("Encoder weights are trainable in NexNet_Seg_ssl.")

        # Encoder
        self.nb_1 = NexBlock(input_shape[2], filters[0], kernel_sizes[0], dilation_rate=2, use_masa=use_masa, gamma=gamma)
        self.pool_112 = nn.MaxPool2d(2)
        self.nb_2 = NexBlock(filters[0], filters[1], kernel_sizes[1], dilation_rate=2, use_masa=use_masa, gamma=gamma)
        self.pool_56 = nn.MaxPool2d(2)
        self.nb_3 = NexBlock(filters[1], filters[2], kernel_sizes[2], dilation_rate=2, use_masa=use_masa, gamma=gamma)
        self.pool_28 = nn.MaxPool2d(2)
        self.nb_4 = NexBlock(filters[2], filters[3], kernel_sizes[3], dilation_rate=2, use_masa=use_masa, gamma=gamma)
        self.pool_14 = nn.MaxPool2d(2)
        self.nb_5 = NexBlock(filters[3], filters[4], kernel_sizes[4], dilation_rate=2, use_masa=use_masa, gamma=gamma)
        self.pool_7 = nn.MaxPool2d(2)
        self.nb_6 = NexBlock(filters[4], filters[5], kernel_sizes[5], dilation_rate=2, use_masa=use_masa, gamma=gamma)

        # Decoder
        # Adjust input channels for up_conv_14 to match the concatenation of nb_6 (filters[5] + encoder_bottleneck_channels) and nb_5_att (filters[4])
        self.up_14 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_14 = DoubleConvMaSA(filters[5] + self.encoder_bottleneck_channels + filters[4], filters[4])  # 512 + 1024 + 256 = 1792
        self.up_28 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_28 = DoubleConvMaSA(filters[4] + filters[3], filters[3])
        self.up_56 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_56 = DoubleConvMaSA(filters[3] + filters[2], filters[2])
        self.up_112 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_112 = DoubleConvMaSA(filters[2] + filters[1], filters[1])
        self.up_224 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_224 = DoubleConvMaSA(filters[1] + filters[0], filters[0], dropout=dr)

        # Final layer (output raw logits)
        self.conv_final = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        nb_1, nb_1_att = self.nb_1(x)
        pool_112 = self.pool_112(nb_1)
        nb_2, nb_2_att = self.nb_2(pool_112)
        pool_56 = self.pool_56(nb_2)
        nb_3, nb_3_att = self.nb_3(pool_56)
        pool_28 = self.pool_28(nb_3)
        nb_4, nb_4_att = self.nb_4(pool_28)
        pool_14 = self.pool_14(nb_4)
        nb_5, nb_5_att = self.nb_5(pool_14)
        pool_7 = self.pool_7(nb_5)

        # Get encoder embedding
        encoder_output = self.encoder.get_embedding(x)
        nb_6, nb_6_att = self.nb_6(pool_7)
        nb_6 = torch.cat([nb_6_att, encoder_output], dim=1)  # Use nb_6_att instead of nb_6

        # Decoder with skip connections
        up_14 = torch.cat([self.up_14(nb_6), nb_5_att], dim=1)
        up_conv_14 = self.up_conv_14(up_14)
        up_28 = torch.cat([self.up_28(up_conv_14), nb_4_att], dim=1)
        up_conv_28 = self.up_conv_28(up_28)
        up_56 = torch.cat([self.up_56(up_conv_28), nb_3_att], dim=1)
        up_conv_56 = self.up_conv_56(up_56)
        up_112 = torch.cat([self.up_112(up_conv_56), nb_2_att], dim=1)
        up_conv_112 = self.up_conv_112(up_112)
        up_224 = torch.cat([self.up_224(up_conv_112), nb_1_att], dim=1)
        up_conv_224 = self.up_conv_224(up_224)

        # Output raw logits
        output = self.conv_final(up_conv_224)
        return output
