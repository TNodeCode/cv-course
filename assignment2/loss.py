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

    L = None

    dS = None

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

    L = None

    dS = None

    ############################################################
    ###                    END OF YOUR CODE                  ###
    ############################################################
    return L, dS


