import numpy as np
from .activation import *



__all__ = ['ReLU_test', 'Sigmoid_test', 'Tanh_test']



# Create some test inputs.
x = np.array([
    [[[-0.01640701, -0.57770197],
      [ 0.37487488, -0.77260795]],
     [[ 0.31117251,  1.28778772],
      [-0.9739217,  -0.15076198]],
     [[ 0.67561644, -0.75053469],
      [-0.68843359,  1.13142207]]],
    [[[-0.48168604,  0.37160233],
      [ 0.41082495, -0.40195236]],
     [[-1.4123589,   2.57995339],
      [-0.20205541,  0.5211701 ]],
     [[ 0.36500362, -1.6598223 ],
      [ 3.39411579, -1.46596421]]]
])

# Create dummy upstream gradient.
out_grad = np.linspace(-0.3, 0.4, num=np.prod(x.shape))
out_grad = np.reshape(out_grad, x.shape)




def ReLU_test():
    """
    Test the ReLU activation function.
    
    Returns:
        - (bool): True if test was passed and False otherwise.

    """
    relu = ReLU()

    # Compute forward pass.
    out = relu(x)

    # Define correct result.
    correct_out = np.array([
        [[[0.,         0.        ],
          [0.37487488, 0.        ]],
         [[0.31117251, 1.28778772],
          [0.,         0.        ]],
         [[0.67561644, 0.        ],
          [0.,         1.13142207]]],
        [[[0.,         0.37160233],
          [0.41082495, 0.        ]],
         [[0.,         2.57995339],
          [0.,         0.5211701 ]],
         [[0.36500362, 0.        ],
          [3.39411579, 0.        ]]]
    ])

    # Check result.
    res_forward = np.allclose(out, correct_out)

    # Compute backward pass.
    in_grad = relu.backward(out_grad)

    # Define correct results.
    correct_in_grad = np.array([
        [[[-0.,         -0.        ],
          [-0.23913043, -0.        ]],
         [[-0.17826087, -0.14782609],
          [-0.,         -0.        ]],
         [[-0.05652174, -0.        ],
          [ 0.,          0.03478261]]],
        [[[ 0.,          0.09565217],
          [ 0.12608696,  0.        ]],
         [[ 0.,          0.2173913 ],
          [ 0.,          0.27826087]],
         [[ 0.30869565,  0.        ],
          [ 0.36956522,  0.        ]]]
    ])

    # Check result.
    res_backward = np.allclose(in_grad, correct_in_grad)

    return res_forward and res_backward



def Sigmoid_test():
    """
    Test the sigmoid activation function.

    Returns:
        - (bool): True if test was passed and False otherwise.

    """
    sigmoid = Sigmoid()

    # Compute forward pass.
    out = sigmoid(x)

    # Define correct result.
    correct_out = np.array([
        [[[0.49589834, 0.35946154],
          [0.59263639, 0.31591522]],
         [[0.57717143, 0.7837725 ],
          [0.27409951, 0.46238073]],
         [[0.66275963, 0.32070481],
          [0.33438162, 0.75610124]]],
        [[[0.38185407, 0.5918461 ],
          [0.60128567, 0.40084335]],
         [[0.19586226, 0.92956022],
          [0.44965731, 0.62742133]],
         [[0.59025113, 0.15978585],
          [0.96752013, 0.18755681]]]
    ])

    # Check result.
    res_forward = np.allclose(out, correct_out)

    # Compute backward pass.
    in_grad = sigmoid.backward(out_grad)

    # Define correct results.
    correct_in_grad = np.array([
        [[[-0.07499495, -0.06206711],
          [-0.05773051, -0.0451018 ]],
         [[-0.0435036,  -0.02505256],
          [-0.02335723, -0.02161607]],
         [[-0.01263313, -0.00568313],
          [ 0.0009677,   0.00641434]]],
        [[[ 0.01539401,  0.02310615],
          [ 0.03022824,  0.03759151]],
         [[ 0.0294457,   0.01423435],
          [ 0.06132843,  0.06504732]],
         [[ 0.0746595,   0.04552973],
          [ 0.01161356,  0.0609517 ]]]
    ])

    # Check result.
    res_backward = np.allclose(in_grad, correct_in_grad)

    return res_forward and res_backward



def Tanh_test():
    """
    Test the tanh activation function.

    Returns:
        - (bool): True if test was passed and False otherwise.

    """
    tanh = Tanh()

    # Compute forward pass.
    out = tanh(x)

    # Define correct result.
    correct_out = np.array([
        [[[-0.01640554, -0.52099317],
          [ 0.35824834, -0.64844337]],
         [[ 0.3015034,   0.85854604],
          [-0.75042259, -0.14963004]],
         [[ 0.58866223, -0.63546783],
          [-0.59697477,  0.8115054 ]]],
        [[[-0.44759289,  0.35539246],
          [ 0.38917291, -0.38161824]],
         [[-0.88799401,  0.9885811 ],
          [-0.19934985,  0.47860259]],
         [[ 0.34961372, -0.93019324],
          [ 0.99774862, -0.89880478]]]
    ])

    # Check result.
    res_forward = np.allclose(out, correct_out)

    # Compute backward pass.
    in_grad = tanh.backward(out_grad)

    # Define correct results.
    correct_in_grad = np.array([
        [[[-0.29991926, -0.19639608],
          [-0.20843999, -0.12094355]],
         [[-0.16205619, -0.03886329],
          [-0.05128426, -0.08500964]],
         [[-0.03693564, -0.01555254],
          [ 0.00279835,  0.01187683]]],
        [[[ 0.05215178,  0.08357094],
          [ 0.10699039,  0.13372709]],
         [[ 0.03953507,  0.00493639],
          [ 0.23797739,  0.21452231]],
         [[ 0.27096386,  0.04569461],
          [ 0.00166219,  0.07685998]]]
    ])

    # Check result.
    res_backward = np.allclose(in_grad, correct_in_grad)

    return res_forward and res_backward


