import numpy as np
from .layer import *



__all__ = ['Vector_test', 'Linear_test', 'Conv2d_test', 'MaxPool_test']



def Vector_test():
    """
    Test the Vector layer.

    Returns:
        - (bool): True if test was passed and False otherwise.

    """
    vec = Vector()

    # Set shape for test inputs and outputs.
    inputs_shape = (2, 3, 4, 5)
    outputs_shape = (2, 60)

    # Create test inputs.
    x = np.zeros(inputs_shape)

    # Create upstream gradient.
    out_grad = np.zeros(outputs_shape)

    # Compute forward pass.
    out = vec(x)

    # Compare shapes
    res_forward = out.shape == outputs_shape

    # Compute backward pass
    in_grad = vec.backward(out_grad)

    # Compare shapes.
    res_backward = in_grad.shape == inputs_shape

    return res_forward and res_backward



def Linear_test():
    """
    Test the Linear layer.
    
    Returns:
        - (bool): True if test was passed and False otherwise.

    """
    num_inputs = 2

    # Set input and output dimensions.
    input_dim = 5
    output_dim = 3

    # Create inputs.
    x = np.linspace(-0.1, 0.5, num=num_inputs * input_dim)
    x = np.reshape(x, (num_inputs, input_dim))

    # Create weights.
    weights = np.linspace(-0.2, 0.3, num=output_dim * input_dim)
    weights = np.reshape(weights, (input_dim, output_dim))

    # Create bias.
    bias = np.linspace(-0.3, 0.1, num=output_dim)

    # Create upstream gradient.
    out_grad = np.linspace(-0.2, 0.4, num=num_inputs * output_dim)
    out_grad = np.reshape(out_grad, (num_inputs, output_dim))

    # Create fully connected layer.
    fc = Linear(input_dim, output_dim)

    # Reset parameters.
    fc.param['weight'] = weights
    fc.param['bias'] = bias

    # Compute forward pass.
    out = fc(x)

    # Define correct outputs.
    correct_out = np.array([
        [-0.22619048, -0.0202381,   0.18571429],
        [-0.20238095,  0.06309524,  0.32857143]
    ])

    # Check result.
    res_forward = np.allclose(out, correct_out)

    # Compute backward pass.
    in_grad = fc.backward(out_grad)

    # Get gradients for parameters.
    weights_grad = fc.grad['weight']
    bias_grad = fc.grad['bias']

    # Compare gradients.
    res_backward = [

        np.allclose(in_grad, np.array([
            [ 0.048,       0.02228571, -0.00342857, -0.02914286, -0.05485714],
            [-0.12942857, -0.03942857,  0.05057143,  0.14057143,  0.23057143]
        ])),

        np.allclose(weights_grad, np.array([
            [0.05733333, 0.07333333, 0.08933333],
            [0.05466667, 0.08666667, 0.11866667],
            [0.052,      0.1,        0.148     ],
            [0.04933333, 0.11333333, 0.17733333],
            [0.04666667, 0.12666667, 0.20666667]
        ])),

        np.allclose(bias_grad, np.array([-0.04, 0.2, 0.44]))
    ]

    return res_forward and all(res_backward)



def Conv2d_test():
    """
    Test the Conv2d layer.
    
    Returns:
        - (bool): True if test was passed and False otherwise.

    """
    inputs_shape  = (2, 2, 3, 3)
    outputs_shape = (2, 2, 2, 2)
    weights_shape = (2, 2, 3, 3)

    # Create inputs.
    x = np.linspace(-0.1, 0.5, num=np.prod(inputs_shape))
    x = np.reshape(x, inputs_shape)

    # Create weights.
    weights = np.linspace(-0.2, 0.3, num=np.prod(weights_shape))
    weights = np.reshape(weights, weights_shape)

    # Create bias.
    bias = np.linspace(-0.1, 0.2, num=2)

    # Create upstream gradient.
    out_grad = np.linspace(-0.2, 0.3, num=np.prod(outputs_shape))
    out_grad = np.reshape(out_grad, outputs_shape)

    # Create conv layer.
    conv = Conv2d(2, 2, kernel_size=3, padding=1, stride=2)

    # Reset parameters.
    conv.param['weight'] = weights
    conv.param['bias'] = bias

    # Compute forward pass.
    out = conv(x)

    # Define correct results.
    correct_out = np.array([
        [[[-0.06,       -0.07012245],
          [-0.10212245, -0.124     ]],
         [[ 0.2635102,   0.28865306],
          [ 0.32718367,  0.34057143]]],
        [[[-0.18342857, -0.22881633],
          [-0.33134694, -0.3884898 ]],
         [[ 0.77485714,  0.76473469],
          [ 0.73273469,  0.71085714]]]
    ])

    # Compare outputs
    res_forward = np.allclose(out, correct_out)

    # Compute backward pass.
    in_grad = conv.backward(out_grad)

    # Get gradients for parameters.
    weights_grad = conv.grad['weight']
    bias_grad = conv.grad['bias']

    # Compare gradients.
    res_backward = [

        np.allclose(in_grad, np.array([
            [[[ 0.02095238,  0.04,        0.02      ],
              [ 0.03428571,  0.0647619,   0.03238095],
              [ 0.01904762,  0.03619048,  0.01809524]],
             [[-0.01333333, -0.02,       -0.00571429],
              [-0.01714286, -0.02095238, -0.00190476],
              [ 0.00190476,  0.01047619,  0.00952381]]],
            [[[ 0.01333333,  0.0247619,   0.01238095],
              [ 0.01904762,  0.03428571,  0.01714286],
              [ 0.01142857,  0.02095238,  0.01047619]],
             [[ 0.04761905,  0.10190476,  0.0552381 ],
              [ 0.1047619,   0.22285714,  0.12      ],
              [ 0.06285714,  0.13238095,  0.07047619]]]
        ])),

        np.allclose(weights_grad, np.array([
            [[[0.04933333, 0.09161905, 0.04114286],
              [0.08914286, 0.16419048, 0.0727619 ],
              [0.03295238, 0.05885714, 0.0247619 ]],
             [[0.05961905, 0.10190476, 0.04114286],
              [0.08914286, 0.14361905, 0.05219048],
              [0.02266667, 0.028,      0.00419048]]],
            [[[0.08209524, 0.15714286, 0.07390476],
              [0.15466667, 0.2952381,  0.13828571],
              [0.06571429, 0.12438095, 0.05752381]],
             [[0.13352381, 0.24971429, 0.11504762],
              [0.23695238, 0.4392381,  0.2       ],
              [0.09657143, 0.17580952, 0.07809524]]]
        ])),

        np.allclose(bias_grad, np.array([-0.13333333, 0.93333333]))

    ]

    return res_forward and all(res_backward)



def MaxPool_test():
    """
    Test the MaxPool layer.
    
    Returns:
        - (bool): True if test was passed and False otherwise.

    """
    inputs_shape  = (2, 2, 4, 4)
    outputs_shape = (2, 2, 2, 2)

    # Create inputs.
    x = np.linspace(-0.3, 0.5, num=np.prod(inputs_shape))
    x = np.reshape(x, inputs_shape)

    # Create upstream gradient.
    out_grad = np.linspace(-0.2, 0.3, num=np.prod(outputs_shape))
    out_grad = np.reshape(out_grad, outputs_shape)

    # Create layer.
    pool = MaxPool(2)

    # Compute forward pass.
    out = pool(x)

    # Define correct output.
    correct_out = np.array([
        [[[-0.23650794, -0.21111111],
          [-0.13492063, -0.10952381]],
         [[-0.03333333, -0.00793651],
          [ 0.06825397,  0.09365079]]],
        [[[ 0.16984127,  0.1952381 ],
          [ 0.27142857,  0.2968254 ]],
         [[ 0.37301587,  0.3984127 ],
          [ 0.47460317,  0.5       ]]]
    ])

    # Compare results
    res_forward = np.allclose(out, correct_out)

    # Compute backward pass.
    in_grad = pool.backward(out_grad)

    # Define correct gradient.
    correct_in_grad = np.array([
        [[[0.,  0.,         0.,  0.        ],
          [0., -0.2,        0., -0.16666667],
          [0.,  0.,         0.,  0.        ],
          [0., -0.13333333, 0., -0.1       ]],
         [[0.,  0.,         0.,  0.        ],
          [0., -0.06666667, 0., -0.03333333],
          [0.,  0.,         0.,  0.        ],
          [0.,  0.,         0.,  0.03333333]]],
        [[[0.,  0.,         0.,  0.        ],
          [0.,  0.06666667, 0.,  0.1       ],
          [0.,  0.,         0.,  0.        ],
          [0.,  0.13333333, 0.,  0.16666667]],
         [[0.,  0.,         0.,  0.        ],
          [0.,  0.2,        0.,  0.23333333],
          [0.,  0.,         0.,  0.        ],
          [0.,  0.26666667, 0.,  0.3       ]]]
    ])

    # Compare results.
    res_backward = np.allclose(in_grad, correct_in_grad)

    return res_forward and res_backward





