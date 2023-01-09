import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Create residual block with two conv layers.

        Parameters:
            - in_channels (int): Number of input channels.
            - out_channels (int): Number of output channels.
            - stride (int): Stride for first convolution.

        """
        super().__init__()
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First coonvolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        
        # First batch normalization
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        
        # ReLU activation function
        self.activation1 = nn.ReLU()
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        
        # Second batch normalization layer
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        
        self.main = nn.Sequential(
            self.conv1,
            self.batchnorm1,
            self.activation1,
            self.conv2,
            self.batchnorm2
        )
        
        # Convolutional layer with 1x1 kernel for dimensionality reduction
        self.conv_dimred = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False
        )
        
        # Identity layer
        self.identity = nn.Identity()
        
        # Batch normalization for skip connection
        self.batchnorm_skip = nn.BatchNorm2d(out_channels)
        
        # Define skip layer
        if in_channels == out_channels and stride == 1:
            self.skip = self.identity
        else:
            self.skip = nn.Sequential(
                self.conv_dimred,
                self.batchnorm_skip
            )

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################


    def forward(self, x):
        """
        Compute the forward pass through the residual block.

        Parameters:
            - x (torch.Tensor): Input.

        Returns:
            - out (torch.tensor): Output.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        out = F.relu(self.main(x) + self.skip(x))

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return out



class ResNet(nn.Module):

    def __init__(self):
        """
        Creates a residual network.
        """
        super().__init__()
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################
        
        # First convolutional layer of ResNet
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        
        # First batch normalization
        self.batchnorm1 = nn.BatchNorm2d(6)
        
        # ReLU activation function
        self.activation1 = nn.ReLU()
        
        # Pooling layer
        self.pool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
        
        # Flatten output feature maps of size 32 to a vector of size 10
        self.linear = nn.Linear(in_features=32, out_features=10)
        
        self.module = nn.Sequential(
            self.conv1,
            self.batchnorm1,
            self.activation1,
            ResidualBlock(6, 16),
            ResidualBlock(16, 24, stride=2),
            ResidualBlock(24, 32, stride=2),
            ResidualBlock(32, 32, stride=2),
            self.pool,
            nn.Flatten(),
            self.linear
        )
        
        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################


    def forward(self, x):
        """
        Compute the forward pass through the network.

        Parameters:
            - x (torch.Tensor): Input.

        Returns:
            - out (torch.Tensor): Output.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################
        
        out = self.module(x)

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return out





