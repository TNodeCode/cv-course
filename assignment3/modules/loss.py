import numpy as np
from .module import Module



__all__ = ['CrossEntropyLoss']



class CrossEntropyLoss(Module):

    def __init__(self, model):
        """
        Create a cross-entropy loss for the given model.

        Parameters:
            - model (Module): Model to compute the loss for.

        """
        super().__init__()

        self.model = model


    def forward(self, outputs, labels):
        """
        Compute the loss for given inputs and labels.

        Stores the probabilities obtained from applying the
        softmax function to the inputs for computing the
        gradient in the backward pass.

        Parameters:
            - outputs (np.array): Scores generated from the model.
            - labels (np.array): Vector with correct classes.

        Returns:
            - loss (float): Loss averaged over inputs.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        loss = None

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return loss


    def backward(self):
        """
        Compute gradient with respect to model parameters.

        Uses probabilities stored in the forward pass to compute
        the local gradient with respect to the inputs, then
        backpropagates the gradient through the model.

        Returns:
            - in_grad: Gradient with respect to the inputs.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        in_grad = None

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        self.model.backward(in_grad)

        return in_grad



