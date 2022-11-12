import numpy as np
from .module import Module



__all__ = ['ReLU', 'Sigmoid', 'Tanh']



class ReLU(Module):

    def forward(self, x):
        """
        Apply the ReLU activation function to input values.

        Inputs:
            - x (np.array): Input features.

        Returns:
            - out (np.array): Output features.

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
        Compute the gradient with respect to the inputs.

        Parameters:
            - out_grad (np.array): Gradient with respect to module output.

        Returns:
            - in_grad (np.array): Gradient with respect to module input.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        in_grad = None

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return in_grad



class Sigmoid(Module):

    def forward(self, x):
        """
        Apply the sigmoid activation function to input values.

        Inputs:
            - x (np.array): Input features.

        Returns:
            - out (np.array): Output features.

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
        Compute the gradient with respect to the inputs.

        Parameters:
            - out_grad (np.array): Gradient with respect to module output.

        Returns:
            - in_grad (np.array): Gradient with respect to module input.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        in_grad = None

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return in_grad



class Tanh(Module):

    def forward(self, x):
        """
        Apply the tanh activation function to input values.

        Inputs:
            - x (np.array): Input features.

        Returns:
            - out (np.array): Output features.

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
        Compute the gradient with respect to the inputs.

        Parameters:
            - out_grad (np.array): Gradient with respect to module output.

        Returns:
            - in_grad (np.array): Gradient with respect to module input.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        in_grad = None

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return in_grad


