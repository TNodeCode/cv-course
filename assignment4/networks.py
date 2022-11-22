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

        # First coonvolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True
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
            stride=stride,
            padding=1,
            bias=True
        )
        
        # Second batch normalization layer
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        
        # Convolutional layer with 1x1 kernel for dimensionality reduction
        self.conv_dimred = nn.Conv2d(
            in_channels=out_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        # Identity layer
        self.identity = nn.Identity()
        
        # Define skip layer
        if in_channels == out_channels:
            self.skip = self.conv_dimred
        else:
            self.skip = self.identity
        
        # Third batch normalization layer
        self.batchnorm3 = nn.BatchNorm2d(1)

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

        # First branch
        r = self.conv1(x)
        r = self.batchnorm1(r)
        r = self.activation1(r)
        r = self.conv2(r)
        r = self.batchnorm2(r)
        
        # Second branch
        x_id = self.skip(x)
        
        # Input plus residual
        out = x_id + r

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
        
        # Numer of residual blocks
        self.n_blocks = 4
        
        # First convolutional layer of ResNet
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        
        # First batch normalization
        self.batchnorm1 = nn.BatchNorm2d(3)
        
        # ReLU activation function
        self.activation1 = nn.ReLU()
        
        # Pooling layer
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
                
        # Flatten output feature maps of size 32x32 to a vector of size 1024x1
        self.linear = nn.Linear(in_features=3*16*16, out_features=10)
        
        # Residual blocks
        self.blocks = []
        
        for i in range(self.n_blocks):
            self.blocks.append(ResidualBlock(3, 3))
        
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

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.activation1(x)
        
        # Run tensor through residual blocks
        for i in range(self.n_blocks):
            x = self.blocks[i](x)        

        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.linear(x)
        
        out = x

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return out





