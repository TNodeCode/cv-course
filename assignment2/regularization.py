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

    # Absolute sum of all weights multiplied by the regularization strength
    # The last row contains the bias components and is not part of the sum
    R = r * abs(W[:-1]).sum()

    # derivatives of the weight components
    weights_derivative = W[:-1] / abs(W[:-1]) * r
    
    # derivatives of the bias components
    bias_derivative = np.zeros((1,W.shape[1]))
                               
    # concatenate weight derivatives and bias derivatives into a single matrix
    dW = np.concatenate([weights_derivative, bias_derivative], axis=0)

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

    R = None

    dW = None

    ############################################################
    ###                   END OF YOUR CODE                   ###
    ############################################################
    return R, dW





