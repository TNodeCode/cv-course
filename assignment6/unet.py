from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import feature_extraction

TensorDict = Dict[str, torch.Tensor]


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        _cnn = models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.IMAGENET1K_V1)

        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "stem": "c0",
                "trunk_output.block1": "c1",
                "trunk_output.block2": "c2",
                "trunk_output.block3": "c3",
                "trunk_output.block4": "c4",
            },
        )

        dummy_out = self.backbone(torch.randn(2, 3, 256, 256))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 256, 256)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"\tShape of {level_name} features: {tuple(feature_shape)}")

    @property
    def out_channels(self) -> List[int]:
        return [32, 32, 64, 160, 400]

    def forward(self, x) -> TensorDict:
        return self.backbone(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        ##################################################################################
        #                                START OF YOUR CODE
        ##################################################################################
        # Fill the conv list with the following layers:
        #   conv -> batchnorm -> relu -> conv -> batchnorm -> relu
        #
        # Use 3x3 convolutions with padding such that the output has the same spatial
        # dimensions as the input. Use the in_channels and out_channels parameters to
        # define the number of input and output channels and use the out_channels to
        # define the intermediate number of channels.
        #
        # Initialize the weights of the convolutions with the Kaiming uniform with fan_out
        # and the correct non-linearity. Initialize the weights of the batchnorm layers
        # with 1 and all biases with 0.
        ##################################################################################
        
        # Add stack of layers to network
        conv = []
        
        # First convolutional layer
        conv.append(
                torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        nn.init.kaiming_uniform_(conv[-1].weight, nonlinearity="relu")
        
        # Batch normalization
        conv.append(torch.nn.BatchNorm2d(out_channels))
        conv[-1].weight.data.fill_(1)
        conv[-1].bias.data.zero_()
        
        # Activation function
        conv.append(torch.nn.ReLU())
        
        # Second convolutional layer
        conv.append(
            torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        nn.init.kaiming_uniform_(conv[-1].weight, nonlinearity="relu")
        
        # Batch Normalization
        conv.append(torch.nn.BatchNorm2d(out_channels))
        conv[-1].weight.data.fill_(1)
        conv[-1].bias.data.zero_()
        
        # Activation function
        conv.append(torch.nn.ReLU())

        ##################################################################################
        #                                END OF YOUR CODE
        ##################################################################################
        self.conv = nn.Sequential(*conv)

    def forward(self, x, encoder_features=None):
        """
        Performs a forward pass of the decoder block. If encoder_features are provided,
        the decoder block will perform an upsampling and concatenation with the
        encoder_features.

        Args:
            x: input tensor
            encoder_features: features from the encoder. If not None, the input tensor
                will be upsampled and concatenated with the encoder features.

        Returns:
            output tensor of shape (batch_size, out_channels, height, width)
        """
        ##################################################################################
        #                                START OF YOUR CODE
        ##################################################################################
        # First upsample the input tensor to have double the spatial dimensions using
        # F.interpolate with mode="nearest". Then concatenate the upsampled input tensor
        # with the encoder_features. Finally, pass the concatenated tensor through the
        # conv layers.
        ##################################################################################

        x = F.interpolate(x, mode="nearest", scale_factor=2)
        x = torch.cat((x, encoder_features), dim=1)
        x = self.conf(x)
        
        ##################################################################################
        #                                END OF YOUR CODE
        ##################################################################################
        return x


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels: List[int],
                 decoder_channels: List[int],
                 ):
        """
        Args:
            encoder_channels: number of channels in the encoder output, ordered from
                image input to the last encoder output.
            decoder_channels: number of channels in the decoder output, ordered from
                the decoder bottom to the network output.
        """
        super().__init__()
        ################################################################################################
        #                                START OF YOUR CODE
        ################################################################################################
        # Create a list of DecoderBlock modules. The number of decoder blocks should be equal to the
        # number of encoder channels. The number of output channels of the decoder blocks is defined
        # by the decoder_channels list. Ensure to implement the skip connections by passing the
        # encoder features to the decoder blocks and that the resulting output of the decoder block
        # has the same spatial dimensions as the input image.

        blocks = []
        
        # Number of decoder blocks (equal to number of encoder channels)
        n_encoder_channels = len(encoder_channels)
        
        # Number of channels of the latest decoder block
        c_dec_latest = 0
        
        # Create stack of decoder blocks
        for i, (c_enc, c_dec) in enumerate(zip(encoder_channels, decoder_channels)):
            blocks.append(
                DecoderBlock(
                    in_channels=encoder_channels[n_encoder_channels - i - 1] + c_dec_latest,
                    out_channels=c_dec
                )
            )
            c_dec_latest = decoder_channels[i]

        ################################################################################################
        #                                END OF YOUR CODE
        ################################################################################################

        self.blocks = nn.ModuleList(blocks)

    def forward(self, features: TensorDict) -> torch.Tensor:
        """
        Performs a forward pass of the decoder.

        Args:
            features: a dictionary of tensors with the following keys 'c0', 'c1', 'c2',
                'c3', 'c4' where the values are the output of the encoder.
        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################
        
        # Layers that are passed from the encoder to the decoder
        keys = ["c4", "c3", "c2", "c1", "c0"]
        
        # First get the output of the c4 layer
        x = features["c4"]
        
        # Run the data and the extracted latent features from the encoder through teh decoder blocks
        for i, key in enumerate(keys):
            x = self.blocks[i](x, None) if key == "c4" else self.blocks[i](x, features[key])

        # Interpolate the resulting tensor so that it has the same spatial dimension as the input image
        x = F.interpolate(x, scale_factor=2, mode="nearest")
            
        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return x


class UNet(nn.Module):
    def __init__(self,
                 num_classes=1,
                 decoder_channels: List[int] = [256, 128, 64, 32, 16],
                 ):
        """
        Unet model.

        Args:
            num_classes: number of classes in the segmentation mask
            decoder_channels: number of channels in the decoder output, ordered from
                the decoder bottom to the network output.

        """
        super().__init__()
        self.num_classes = num_classes

        self.encoder = Encoder()
        self.decoder = Decoder(self.encoder.out_channels, decoder_channels)

        # The final output of the network is a convolution with num_classes output channels
        self.head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        features = self.encoder(x)
        x = self.decoder(features)
        x = self.head(x)
        return x
