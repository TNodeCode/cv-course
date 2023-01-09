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
        
        # Store outputs and labels for backpropagation
        self.outputs = outputs
        self.labels = labels

        # Create a mask that extracts only the score values for the true classes from the score matrix
        mask_correct = np.zeros((outputs.shape[0], outputs.shape[1]))
        mask_correct[np.arange(0, outputs.shape[0]), labels] = 1
        mask_correct = mask_correct.astype(bool)

        # Get the maximum value of the score matrix for each row
        max_exp = np.max(outputs, axis=1).reshape(-1, 1)

        # Shift the score matrix values by the maximum value of each row to the left
        outputs = outputs - max_exp

        # Calculate the nominator and the denominator
        denominator = np.exp(outputs).sum(axis=1)
        nominator = np.exp(outputs[mask_correct])

        # Calculate the fraction
        fraction = np.exp(outputs[mask_correct]) / np.exp(outputs).sum(axis=1)

        # Add a small number to fractions that are zero, because the logarithm of zero is not defined
        fraction[fraction == 0] = 1e-6

        # calculate the loss for a single record
        Ls = - np.log(fraction)

        # calculate the total loss
        loss = Ls.sum() / outputs.shape[0]
    
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

        # Get outputs and labels
        outputs = self.outputs
        labels = self.labels
        
        # Create a mask that extracts only the score values for the true classes from the score matrix
        mask_correct = np.zeros((outputs.shape[0], outputs.shape[1]))
        mask_correct[np.arange(0, outputs.shape[0]), labels] = 1
        mask_correct = mask_correct.astype(bool)

        # Get the maximum value of the score matrix for each row
        max_exp = np.max(outputs, axis=1).reshape(-1, 1)

        # Shift the score matrix values by the maximum value of each row to the left
        outputs = outputs - max_exp

        # Calculate the denominator
        denominator = np.exp(outputs).sum(axis=1)

        # normalize all matrix components with the sigmoid function
        sigmoid = np.exp(outputs) / denominator.reshape(-1, 1)

        # For all correct predictions we have to subtract one from the sigmoid function output
        sigmoid[mask_correct] -= 1

        # Finally we have to devide the matrix by the number of records
        in_grad = sigmoid / outputs.shape[0]
        
        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        self.model.backward(in_grad)

        return in_grad



