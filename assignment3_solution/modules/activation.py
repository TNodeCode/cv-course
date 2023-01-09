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

        # Make a copy of the input vector and replace all values that
        # are smaller than zero with zero
        self.input = x
        out = x.copy()
        out[out <= 0] = 0

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

        # Make a copy of the inpput vector and set the values to 1
        # if the argument is greater than zero and to 0 otherwise
        in_grad = np.where(self.input <= 0, 0, 1) * out_grad

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

        self.input = x
        out = 1 / (1 + np.exp(-x))

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

        in_grad = (1 / (1 + np.exp(-self.input))) * (1 - (1 / (1 + np.exp(-self.input)))) * out_grad

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

        self.input = x
        out = (np.exp(self.input) - np.exp(-self.input)) / (np.exp(self.input) + np.exp(-self.input))

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

        in_grad = (1 - np.tanh(self.input)**2) * out_grad

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return in_grad


