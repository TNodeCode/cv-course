import numpy as np



def L1_reg(W, r):
    """
    Compute L1 regularization loss for weights in the given parameter matrix.

    The last row in W is assumed to be the bias.
    Regularization is only applied to the weights and not to the bias.

    Parameters:
        - W: Parameter matrix with shape (D+1, K) with input dimension D and number of classes K.
        - r: Regularization strength.

    Returns:
        - R: Regularization loss.
        - dW: Partial derivatives of L with respect to W.

    """
    ############################################################
    ###                  START OF YOUR CODE                  ###
    ############################################################

    # The loss is the sum of the absolute values of the weights
    R = r * np.sum(np.abs(W[:-1]))

    # Partial derivatives
    dW = r * np.sign(W)
                               
    # Derivative for bias is zero
    dW[-1] = 0

    ############################################################
    ###                   END OF YOUR CODE                   ###
    ############################################################
    return R, dW



def L2_reg(W, r):
    """
    Compute L2 regularization loss for weights in the given parameter matrix.

    The last row in W is assumed to be the bias.
    Regularization is only applied to the weights and not to the bias.

    Parameters:
        - W: Parameter matrix with shape (D+1, K) with input dimension D and number of classes K.
        - r: Regularization strength.

    Returns:
        - R: Regularization loss.
        - dW: Partial derivatives of L with respect to W.

    """
    ############################################################
    ###                  START OF YOUR CODE                  ###
    ############################################################

    # Absolute sum of all squared weights multiplied by the regularization strength
    # The last row contains the bias components and is not part of the sum
    R = r * ((W[:-1]) ** 2).sum()

    # derivatives of the weight components
    dW = 2 * r * W
    
    # derivatives of the bias components
    dW[-1] = 0
    
    ############################################################
    ###                   END OF YOUR CODE                   ###
    ############################################################
    return R, dW





