import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO remove
def log(*args):
    pass
    #print(*args)



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
            in_channels=in_channels,
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
        log("x before conv1", x.shape)
        r = self.conv1(x)
        log("r before batchnorm1", r.shape)
        r = self.batchnorm1(r)
        log("r before activation1", r.shape)
        r = self.activation1(r)
        log("r before conv2", r.shape)
        r = self.conv2(r)
        log("r before batchnorm2", r.shape)
        r = self.batchnorm2(r)
        log("r after batchnorm2", r.shape)
        
        # Second branch
        if x.shape[1] > 1:
            log("x before conv_dimred", x.shape)
            x_id = self.conv_dimred(x)
            log("x_id after conv_dimred", x_id.shape)
        else:
            x_id = self.identity(x)
        log("x_id before batchnorm3", x_id.shape)
        x_id = self.batchnorm3(x_id)
        log("x_id after batchnorm3", x_id.shape)
        
        # Input plus residual
        out = x_id + r
        log("out tensor", out.shape)

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
            bias=False
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

        log("x before conv1", x.shape)
        x = self.conv1(x)
        log("x before batchnorm1", x.shape)
        x = self.batchnorm1(x)
        log("x before activation", x.shape)
        x = self.activation1(x)
        
        log("x before block", x.shape)
        # Run tensor through residual blocks
        for i in range(self.n_blocks):
            log("BLOCK i", i)
            x = self.blocks[i](x)        

        log("x before pooling", x.shape)
        x = self.pool(x)
        log("x before flatten", x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        log("x before linear", x.shape)
        x = self.linear(x)
        log("x out", x.shape)
        
        out = x

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return out





