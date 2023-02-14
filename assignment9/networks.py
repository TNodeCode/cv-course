import torch
import torch.nn as nn
import torch.nn.functional as F



__all__ = ['Discriminator', 'Generator']



def init_params(module):
    """
    Initialize parameters in convolution and batchnorm layers.

    Parameters:
        - module (nn.Module): Layer with parameters.

    """
    ############################################################
    ###                  START OF YOUR CODE                  ###
    ############################################################

    for m in module.modules():
        # Check for the module type
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # Draw weights from a normal distribution with mean 0.0 and standard deviation 0.02
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d)):
            # Draw weights from a normal distribution with mean 0.0 and standard deviation 0.02
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            # Init bias with zero
            nn.init.constant_(m.bias.data, 0)


    ############################################################
    ###                   END OF YOUR CODE                   ###
    ############################################################



class Discriminator(nn.Module):

    def __init__(self, channels, depth):
        """
        Create discriminator network.

        Parameters:
            - channels (int): Number of output channels.
            - depth (int): Depth of intermediate feature maps.

        """
        super().__init__()
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        # Hyperparameters
        self.channels = channels
        self.depth = depth
        self.kernel_size_softconv = 4
        
        # Layers
        conv_blocks = []
        
        conv_blocks.append(self._block_start(in_channels=channels, out_channels=2*channels))
        
        for i in range(1, depth):
            in_channels = (2**i)*channels
            out_channels = 2*in_channels
            conv_blocks.append(self._block(in_channels=in_channels, out_channels=out_channels))
        
        self.conv = nn.Sequential(*conv_blocks)
        
        self.softconf = nn.Conv2d(
            in_channels=2**depth,
            out_channels=1,
            kernel_size=2, 
            stride=1,
            padding=1,
            bias=False
        )
        
        conv_out_dim = ((28 // 2**(depth)) - self.kernel_size_softconv + 1)**2
        self.fc1 = nn.Linear(conv_out_dim, conv_out_dim)
        self.fc2 = nn.Linear(conv_out_dim, conv_out_dim)
        self.fc3 = nn.Linear(conv_out_dim, 1)

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        self.apply(init_params)
        
    def _block_start(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        """
        Compute forward pass through network.

        Parameters:
            - x (torch.Tensor): Input images.

        Returns:
            - p (torch.Tensor): Probability of being real.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        # Run inputs through convolutional layers
        x = self.conv(x)
        # Flatten output of convolutional block to a single channel with a 1x1 convolution
        x = self.softconf(x)
        # Fatten tensor
        x = torch.flatten(x, start_dim=1)
        # Run inputs through linear layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Make final prediction
        p = F.sigmoid(self.fc3(x))

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return p



class Generator(nn.Module):

    def __init__(self, latent_dim, channels, depth):
        """
        Create generator network.

        Parameters:
            - latent_dim (int): Dimension of latent vectors.
            - channels (int): Number of output channels.
            - depth (int): Depth of intermediate feature maps.

        """
        super().__init__()
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        # Hyperparameters
        self.latent_dim = latent_dim
        self.channels = channels
        self.depth = depth
        self.generator_features = 7
        
        # Layers
        layer_channels = [latent_dim] + [(depth-i)*self.generator_features for i in range(depth)]
        conv_blocks = []
        
        for i in range(len(layer_channels) - 1):
            in_channels=layer_channels[i]
            out_channels=layer_channels[i+1]
            conv_blocks.append(self._block(in_channels, out_channels))
        
        self.conv = nn.Sequential(*conv_blocks)   
        
        self.out = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=layer_channels[-1],
                out_channels=channels,
                kernel_size=4,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.Tanh()
        )

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        self.apply(init_params)
        
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )


    def forward(self, z):
        """
        Compute forward pass through network.

        Parameters:
            - z (torch.Tensor): Latent vectors.

        Returns:
            - x (torch.Tensor): Fake images.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################
        
        x = self.conv(z)
        x = self.out(x)

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return x




