import numpy as np



def SVM_loss(S, y, d=1):
    """
    Compute multiclass SVM loss and derivative for a minibatch of scores.

    Parameters:
        - S: Matrix of scores with shape (N, K) with number of samples N and classes K.
        - y: Vector with ground truth labels that has length N.
        - d: Margin hyperparameter.

    Returns:
        - L: Total loss.
        - dS: Partial derivative of L with respect to S.

    """
    ############################################################
    ###                  START OF YOUR CODE                  ###
    ############################################################
    
    # Create a mask that extracts only the score values for the true classes from the score matrix
    mask_correct = np.zeros((S.shape[0], S.shape[1]))
    mask_correct[np.arange(0,S.shape[0]), y] = 1
    mask_correct = mask_correct.astype(bool)

    # Extract score values for correct labels from S
    s_y = S[mask_correct].reshape(-1, 1)
    
    # Extract score values for incorrect labels from S
    s_k = S[~mask_correct].reshape((S.shape[0], S.shape[1] - 1))
    
    # Calculate term that will be compared to zero
    loss_term_incorrect = s_k - s_y + d
    
    # Check if calculated terms are smaller than zero
    mask_smaller = loss_term_incorrect < 0
    
    # Set all components that are smaller than zero to zero
    loss = loss_term_incorrect.copy()
    loss[mask_smaller] = 0
    
    # Calculate loss value for each item
    loss = loss.sum(axis=1).reshape(-1, 1)
    
    # Calculate loss value for whole dataset
    L = (1 / S.shape[0]) * loss.sum()
    
    # Calculate derivatives for incorrect predictions
    dS_incorrect = (loss_term_incorrect > 0).astype(np.float).reshape(-1)
    
    # Calculate derivatives for correct predictions
    dS_correct = - dS_incorrect.reshape((S.shape[0], S.shape[1] - 1)).sum(axis=1)
    
    # Build the derivation matrix
    dS = np.zeros((S.shape[0], S.shape[1]))
    dS[mask_correct] = dS_correct
    dS[~mask_correct] = dS_incorrect
    dS = dS / S.shape[0]

    ############################################################
    ###                    END OF YOUR CODE                  ###
    ############################################################
    return L, dS


def cross_entropy_loss(S, y):
    """
    Compute cross-entropy loss and derivative for a minibatch of scores.

    Parameters:
        - S: Matrix of scores with shape (N, K) with number of samples N and classes K.
        - y: Vector with ground truth labels that has length N.

    Returns:
        - L: Total loss.
        - dS: Partial derivative of L with respect to S.

    """
    ############################################################
    ###                  START OF YOUR CODE                  ###
    ############################################################
    
    # Create a mask that extracts only the score values for the true classes from the score matrix
    mask_correct = np.zeros((S.shape[0], S.shape[1]))
    mask_correct[np.arange(0,S.shape[0]), y] = 1
    mask_correct = mask_correct.astype(bool)
    
    # Get the maximum value of the score matrix for each row
    max_exp = np.max(S, axis=1).reshape(-1, 1)

    # Shift the score matrix values by the maximum value of each row to the left
    S = S - max_exp

    # Calculate the nominator and the denominator
    denominator = np.exp(S).sum(axis=1)
    nominator = np.exp(S[mask_correct])
    
    # Calculate the fraction
    fraction = np.exp(S[mask_correct]) / np.exp(S).sum(axis=1)
    
    # Add a small number to fractions that are zero, because the logarithm of zero is not defined
    fraction[fraction == 0] = 1e-6
    
    # calculate the loss for a single record
    Ls = - np.log(fraction)

    # calculate the total loss
    L = Ls.sum() / S.shape[0]
    
    # normalize all matrix components with the sigmoid function
    sigmoid = np.exp(S) / denominator.reshape(-1, 1)
        
    # For all correct predictions we have to subtract one from the sigmoid function output
    sigmoid[mask_correct] -= 1
    
    # Finally we have to devide the matrix by the number of records
    dS = sigmoid / S.shape[0]

    ############################################################
    ###                    END OF YOUR CODE                  ###
    ############################################################
    return L, dS


