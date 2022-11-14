import numpy as np
from .module import Module



__all__ = ['Vector', 'Linear', 'Conv2d', 'MaxPool']



class Vector(Module):
    
    def forward(self, inputs):
        """
        Converts tensor inputs into vectors.

        Inputs:
            -inputs: Array with shape (N, D1, ..., Dk)

        Returns:
            -outputs: Array with shape (N, D)

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        # Calculate the shape o fthe output vector
        N, D = inputs.shape[0], np.array(inputs.shape[1:]).prod()
        
        # Store original shape
        self.original_shape = inputs.shape
        
        # Reshape vector
        out = inputs.reshape((N, D))

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return out


    def backward(self, out_grad):
        """
        Converts vector inputs into tensors.

        Parameters:
            - out_grad (np.array): Gradient array with shape (N, D).
        
        Returns:
            - in_grad (np.array): Gradient array with shape (N, D1, ..., Dk).

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        in_grad = out_grad.reshape(self.original_shape)

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return in_grad




class Linear(Module):

    def __init__(self, in_features, out_features, bias=True):
        """
        Create linear or affine transformation layer.

        The models weights and bias are stored in the `param`
        dictionary inherited from the Module base class, using
        the keys `weight` and `bias`, respectively.

        Parameters:
            - in_features (int): Dimension of inputs.
            - out_features (int): Dimension of outputs.
            - bias (bool): Use bias or not.

        """
        super().__init__()
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        # Set flag for affine transformation
        self.use_affine_transformation = bias
        
        # Initialize weights
        k_sqrt = np.sqrt(1 / in_features)
        self.param['weight'] = np.random.uniform(-k_sqrt, k_sqrt, size=(in_features, out_features))
        
        # Initialize bias
        if bias:
            self.param['bias'] = np.random.uniform(-k_sqrt, k_sqrt, size=out_features)
        
        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################


    def forward(self, x):
        """
        Compute linear or affine transformation of the inputs.

        Parameters:
            - x (np.array): Inputs with shape (num_samples, in_features).

        Returns:
            - out (np.array): Outputs with shape (num_samples, out_features).

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        # store inputs for gradient calculation
        self.input = x        
        
        # Calculate linear / affine transformation
        out = x @ self.param['weight']
        
        if self.use_affine_transformation:
            out += self.param['bias']
        
        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return out


    def backward(self, out_grad):
        """
        Compute gradient with respect to parameters and inputs.

        The gradient with respect to the weights and bias is stored
        in the `grad` dictionary created by the base class, with
        the keys `weight` and `bias`, respectively.

        Parameters:
            - out_grad (np.array): Gradient with respect to layer output.

        Returns:
            - in_grad (np.array): Gradient with respect to layer input.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        self.grad['weight'] = self.input.T @ out_grad
        
        if self.use_affine_transformation:
            self.grad['bias'] = np.sum(out_grad, axis=0)
        
        in_grad = out_grad @ self.param['weight'].T

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return in_grad



class Conv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True):
        """
        Create a convolutional layer using the given parameters.

        Each filter has the same number of channels as the input and
        the number of filters is equal to the number of output channels.
        If requested, a bias is added to each output unit, with one bias
        value per output channel. Parameters are stored in the `param`
        dictionary with keys `weight` and `bias`, respectively.

        Parameters:
            - in_channels (int): Number of input channels.
            - out_channels (int): Number of output channels.
            - kernel_size (int): Size of filter kernel which is assumed to be square.
            - padding (int): Number of zeros added to borders of input.
            - stride (int): Step size for the filter.
            - bias (bool): Use bias or not.

        """
        super().__init__()
        
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        # Set flags for convolution operation
        self.padding = padding
        self.stride = stride
        self.bias = bias
        
        # Initialize weights
        self.k = 1 / (in_channels * (kernel_size**2))
        k_sqrt = np.sqrt(self.k)
        self.param['weight'] = np.random.uniform(-k_sqrt, k_sqrt,
                                                 (out_channels, in_channels,
                                                  kernel_size, kernel_size))

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################


    def forward(self, x):
        """
        Compute the forward pass through the layer.

        Parameters:
            - x (np.array): Inputs with shape (num_samples, in_channels, in_height, in_width).

        Returns:
            - out (np.array): Outputs with shape (num_samples, out_channels, out_height, out_width).

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        # Save inputs
        self.inputs = x

        # Extract dimensions of input
        dim_x1, dim_x2 = x.shape[2:]
        
        # Extract dimensions of filter
        dim_w1, dim_w2 = self.param['weight'].shape[2:]

        # Calculate output dimensions
        dim_o1 = (dim_x1 + 2 * self.padding - dim_w1) // self.stride + 1
        dim_o2 = (dim_x2 + 2 * self.padding - dim_w2) // self.stride + 1
        
        # Extract number of channels and samples
        n_channels = self.param['weight'].shape[0]
        n_samples = x.shape[0]

        # Create empty tensor for output
        out = np.empty((n_samples, n_channels, dim_o1, dim_o2), x.dtype)

        # Apply padding only to spatial dimensions of input
        x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        # Slice window and apply filter
        for i in range(n_samples):
            # Select i-th sample
            sample = x[i, :, :, :]
            
            # Iterate over output pixels
            for h in range(dim_o1):
                for w in range(dim_o2):
                    # Compute start pixel (h, w)
                    k = h * self.stride
                    l = w * self.stride
                    
                    # Iterate over channels
                    for c in range(n_channels):
                        # Select window
                        window = sample[:, k:k + dim_w1, l:l + dim_w2]
                        
                        # Select channel and bias
                        channel = self.param['weight'][c, :, :, :]
                        bias = self.param['bias'][c]
                        
                        # Compute output pixel (h, w) for i-th sample and channel c
                        out[i, c, h, w] = np.sum(window * channel) + bias

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return out


    def backward(self, out_grad):
        """
        Compute gradient with respect to parameters and inputs.

        The gradient with respect to the weights and bias is stored
        in the `grad` dictionary created by the base class, with
        the keys `weight` and `bias`, respectively.

        Parameters:
            - out_grad (np.array): Gradient with respect to layer output.

        Returns:
            - in_grad (np.array): Gradient with respect to layer input.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        # Get Input dimensions
        n_samples, n_channels, dim_x1, dim_x2 = out_grad.shape
        kernel_size = self.param['weight'].shape[3]
        
        # Pad borders of image with zeros so that a stride greater than one is possible
        X = np.pad(
            self.inputs,
            (
              (0, 0), (0, 0),
              (self.padding, self.padding),
              (self.padding, self.padding)
            )
        )
        
        dX = np.zeros_like(X)
        in_grad = np.zeros_like(self.inputs)
        
        self.grad['bias'] = np.zeros_like(self.param['bias'])
        self.grad['weight'] = np.zeros_like(self.param['weight'])
        
        for i in range(n_samples):
            # Select i-th sample
            X_i = X[i, :, :, :]
            dX_i = dX[i, :, :, :]
            
            # Iterate over height and width
            for h in range(dim_x1):
                for w in range(dim_x2):
                    # Calculate start pixel (k,l)
                    k = h * self.stride
                    l = w * self.stride
                    
                    # Iterate over channels
                    for c in range(n_channels):
                        # Select channel and window
                        channel = self.param['weight'][c, :, :, :]
                        window = X_i[:, k:k+kernel_size, l:l+kernel_size]
                        
                        # Compute weight gradient
                        self.grad['weight'][c, :, :, :] += window * out_grad[i, c, h, w]
                        
                        dX_i[:, k:k+kernel_size, l:l+kernel_size] += channel * out_grad[i, c, h, w]
                        
                        # Compute bias gradient
                        self.grad['bias'][c] += out_grad[i, c, h, w]

                        in_grad[i, :, :, :] = dX_i[:, self.padding:-self.padding, self.padding:-self.padding] # remove padding for output

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return in_grad



class MaxPool(Module):
    
    def __init__(self, kernel_size, stride=None):
        """
        Create max pooling layer with given kernel size and stride.

        If no stride is provided, the stride is set to the kernel
        size, such that non-overlapping areas are filtered for
        the maximum.

        Parameters:
            - kernel_size (int): Size of the pooling region which is assumed to be square.
            - stride (int): Step size of the filter operation.

        """
        super().__init__()
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################


    def forward(self, x):
        """
        Apply max pooling to the given inputs.

        Parameters:
            - x (np.array): Inputs with shape (num_samples, num_channels, in_height, in_width).
        
        Returns:
            - out (np.array): Outputs with shape (num_samples, num_channels, out_height, out_width).

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        # Save input tensor
        self.inputs = x

        # Get input dimensions
        dim_x1, dim_x2 = x.shape[2:]

        # Compute output dimensions
        dim_o1 = (dim_x1 - self.kernel_size) // self.stride + 1
        dim_o2 = (dim_x2 - self.kernel_size) // self.stride + 1
        
        # Extract number of samples and channels
        n_samples = x.shape[0]
        n_channels = x.shape[1]

        # Create empty tensor which has the same shape as the input tensor
        out = np.empty((n_samples, n_channels, dim_o1, dim_o2), x.dtype)

        # Iterate over samples and sample channels
        for i in range(n_samples):
            for c in range(n_channels):
                # Get channel c of i-th sample
                sample = x[i, c, :, :]
                
                # Slide window over image
                for h in range(dim_o1):
                    for w in range(dim_o2):
                        # Compute upper left pixel (k,l) of sliding window
                        k = h * self.stride
                        l = w * self.stride
                        
                        # Get window
                        window = sample[k:k+self.kernel_size, l:l+self.kernel_size]
                        
                        # Extract maximum value of window and save it in output tensor
                        max_val = np.max(window)
                        out[i, c, h, w] = max_val

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return out


    def backward(self, out_grad):
        """
        Compute the gradient with respect to the layer input.

        Parameters:
            - out_grad (np.array): Gradient with respect to layer output.

        Returns:
            - in_grad (np.array): Gradient with respect to layer input.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        # Get Input dimensions
        n_samples, n_channels, dim_x1, dim_x2 = out_grad.shape

        # Create empty tensor which has the same shape as the input tensor
        in_grad = np.zeros_like(self.inputs)

        # Iterate over samples and sample channels
        for i in range(n_samples):
            for c in range(n_channels):
                # Get channel c of i-th sample
                sample = self.inputs[i, c, :, :]
                
                # Slide window over image
                for h in range(dim_x1):
                    for w in range(dim_x2):
                        # Compute upper left pixel (k,l) of sliding window
                        k = h * self.stride
                        l = w * self.stride
                        
                        # Select sliding window
                        window = sample[k:k+self.kernel_size, l:l+self.kernel_size]
                        
                        # Calculate output value
                        mask = window == np.max(window)
                        in_grad[i, c, k:k+self.kernel_size, l:l+self.kernel_size] += mask * out_grad[i, c, h, w]


        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return in_grad



