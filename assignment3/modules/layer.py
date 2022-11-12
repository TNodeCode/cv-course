import numpy as np
from .module import Module



__all__ = ['Vector', 'Linear', 'Conv2d', 'MaxPool']



class Vector(Module):
    
    def forward(self, inputs):
        """
        Converts tensor inputs into vectors.

        Inputs:
            -inputs: Array with shape (N, D1, ..., Dk)

        Returns:
            -outputs: Array with shape (N, D)

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        out = None

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return out


    def backward(self, out_grad):
        """
        Converts vector inputs into tensors.

        Parameters:
            - out_grad (np.array): Gradient array with shape (N, D).
        
        Returns:
            - in_grad (np.array): Gradient array with shape (N, D1, ..., Dk).

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        in_grad = None

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return in_grad




class Linear(Module):

    def __init__(self, in_features, out_features, bias=True):
        """
        Create linear or affine transformation layer.

        The models weights and bias are stored in the `param`
        dictionary inherited from the Module base class, using
        the keys `weight` and `bias`, respectively.

        Parameters:
            - in_features (int): Dimension of inputs.
            - out_features (int): Dimension of outputs.
            - bias (bool): Use bias or not.

        """
        super().__init__()
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################



        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################


    def forward(self, x):
        """
        Compute linear or affine transformation of the inputs.

        Parameters:
            - x (np.array): Inputs with shape (num_samples, in_features).

        Returns:
            - out (np.array): Outputs with shape (num_samples, out_features).

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        out = None

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return out


    def backward(self, out_grad):
        """
        Compute gradient with respect to parameters and inputs.

        The gradient with respect to the weights and bias is stored
        in the `grad` dictionary created by the base class, with
        the keys `weight` and `bias`, respectively.

        Parameters:
            - out_grad (np.array): Gradient with respect to layer output.

        Returns:
            - in_grad (np.array): Gradient with respect to layer input.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        in_grad = None

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return in_grad



class Conv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True):
        """
        Create a convolutional layer using the given parameters.

        Each filter has the same number of channels as the input and
        the number of filters is equal to the number of output channels.
        If requested, a bias is added to each output unit, with one bias
        value per output channel. Parameters are stored in the `param`
        dictionary with keys `weight` and `bias`, respectively.

        Parameters:
            - in_channels (int): Number of input channels.
            - out_channels (int): Number of output channels.
            - kernel_size (int): Size of filter kernel which is assumed to be square.
            - padding (int): Number of zeros added to borders of input.
            - stride (int): Step size for the filter.
            - bias (bool): Use bias or not.

        """
        super().__init__()
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################



        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################


    def forward(self, x):
        """
        Compute the forward pass through the layer.

        Parameters:
            - x (np.array): Inputs with shape (num_samples, in_channels, in_height, in_width).

        Returns:
            - out (np.array): Outputs with shape (num_samples, out_channels, out_height, out_width).

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        out = None

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return out


    def backward(self, out_grad):
        """
        Compute gradient with respect to parameters and inputs.

        The gradient with respect to the weights and bias is stored
        in the `grad` dictionary created by the base class, with
        the keys `weight` and `bias`, respectively.

        Parameters:
            - out_grad (np.array): Gradient with respect to layer output.

        Returns:
            - in_grad (np.array): Gradient with respect to layer input.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        in_grad = None

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return in_grad



class MaxPool(Module):
    
    def __init__(self, kernel_size, stride=None):
        """
        Create max pooling layer with given kernel size and stride.

        If no stride is provided, the stride is set to the kernel
        size, such that non-overlapping areas are filtered for
        the maximum.

        Parameters:
            - kernel_size (int): Size of the pooling region which is assumed to be square.
            - stride (int): Step size of the filter operation.

        """
        super().__init__()
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################



        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################


    def forward(self, x):
        """
        Apply max pooling to the given inputs.

        Parameters:
            - x (np.array): Inputs with shape (num_samples, num_channels, in_height, in_width).
        
        Returns:
            - out (np.array): Outputs with shape (num_samples, num_channels, out_height, out_width).

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        out = None

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return out


    def backward(self, out_grad):
        """
        Compute the gradient with respect to the layer input.

        Parameters:
            - out_grad (np.array): Gradient with respect to layer output.

        Returns:
            - in_grad (np.array): Gradient with respect to layer input.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        in_grad = None

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return in_grad



