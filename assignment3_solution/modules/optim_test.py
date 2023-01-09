import numpy as np
from .layer import Linear
from .optim import *



__all__ = ['SGD_test']



def SGD_test(use_weight_decay=False):
    """
    Test the SGD optimizer.

    Parameters:
        - use_weight_decay (bool): Flag to test weight decay implementation.

    Returns:
        - (bool): True if test was passed and False otherwise.

    """
    shape = (3, 2)

    # Create fully connected linear layer.
    fc = Linear(*shape, bias=False)

    # Create weights.
    weights = np.linspace(-0.3, 0.5, num=np.prod(shape))
    weights = np.reshape(weights, shape)

    # Add weights to layer.
    fc.param['weight'] = weights

    # Create gradient.
    grad = np.linspace(-0.2, 0.3, num=np.prod(shape))
    grad = np.reshape(grad, shape)

    # Add fake gradient to layer.
    fc.grad['weight'] = grad

    # Create optimizer.
    optimizer = SGD(fc, lr=0.1, momentum=0.5, weight_decay=5e-4 if use_weight_decay else 0)

    # Create velocity.
    velocity = np.linspace(-0.1, 0.4, num=np.prod(shape))
    velocity = np.reshape(velocity, shape)

    # Add velocity to optimizer.
    optimizer.targets[fc]['weight'] = velocity

    # Compute update step.
    optimizer.step()

    # Read updated weights.
    weights = fc.param['weight']

    # Define correct result.
    if use_weight_decay:
        correct_weights = np.array([
            [-0.229985, -0.129993],
            [-0.030001,  0.069991],
            [ 0.169983,  0.269975]
        ])
    else:
        correct_weights = np.array([
            [-0.23, -0.13],
            [-0.03,  0.07],
            [ 0.17,  0.27]
        ])

    # Compare results.
    res = np.allclose(weights, correct_weights)

    return res



