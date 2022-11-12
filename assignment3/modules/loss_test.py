import numpy as np
from .module import Module
from .loss import *



__all__ = ['CrossEntropyLoss_test']



def CrossEntropyLoss_test():
    """
    Test the CrossEntropyLoss function.

    Returns:
        - (bool): True if test was passed and False otherwise.

    """
    class Net(Module):
        pass

    # Create dummy network.
    model = Net()

    # Create some fake model outputs.
    outputs = np.array([
        [ 1.7, -2.5, -5.3, 4.1],
        [-1.2,  1.8,  2.0, 2.5],
        [ 2.3, -1.7, -0.4, 0.6]
    ])

    # Create labels.
    labels = np.array([0, 3, 2])

    # Create loss.
    loss = CrossEntropyLoss(model)

    # Compute test loss.
    test_loss = loss(outputs, labels)

    # Define correct outcome.
    correct_loss = 2.060289248015619

    # Compare results.
    res_forward = abs(test_loss - correct_loss) < 1e-5

    # Compute backward pass.
    in_grad = loss.backward()

    # Define correct gradient.
    correct_in_grad = np.array([
        [-3.05645734e-01,  4.15191527e-04,  2.52478228e-05,  3.05205294e-01],
        [ 3.87302498e-03,  7.77917862e-02,  9.50151022e-02, -1.76679913e-01],
        [ 2.62838751e-01,  4.81405965e-03, -3.15669120e-01,  4.80163093e-02]
    ])

    # Compare results.
    res_backward = np.allclose(in_grad, correct_in_grad)

    return res_forward and res_backward



