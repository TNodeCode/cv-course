import numpy as np



__all__ = ['SGD']



class SGD:
    
    def __init__(self, model, lr, momentum=0, weight_decay=0):
        """
        Create optimizer for gradient descent with momentum.

        Extends standard SGD with momentum, so that in each step
        a weighted average of the past gradients is used for the
        update. Setting momentum to zero is equivalent to use
        standard SGD.

        Parameters:
            - model (Module): Network composed of modules.
            - learning_rate (float): Step size to use for parameter updates.
            - momentum (float): Decay rate for past gradients.
            - weight_decay (float): Regularization strength.

        """
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum

        # Collect all modules with parameters.
        modules = [module for module in model.get_modules() if module.param]

        # Store optimization targets.
        self.targets = {}

        for module in modules:
            velocity = {}

            # Create zero initialized velocity array for each parameter.
            for key, value in module.param.items():
                velocity[key] = np.zeros_like(value)

            self.targets[module] = velocity


    def step(self):
        """
        Perform updates for all parameters of the model.

        Stores the velocity for the current time step, computed from
        the most recent gradient and the gradient history weighted
        with the momentum.
        """
        for layer, velocity in self.targets.items():
            for key in layer.param:
                ############################################################
                ###                  START OF YOUR CODE                  ###
                ############################################################

                # Get velocity, weights and gradient
                v = velocity[key]
                param = layer.param[key]
                grad = layer.grad[key]
                
                # Calculate velocity for current timestamp
                v_t = self.momentum * v + self.lr * grad
                
                # Apply regularization
                if self.weight_decay:
                    v_t += self.lr * self.weight_decay * param
                
                # Update velocity and weights
                param -= v_t
                velocity[key] = v_t
                
                ############################################################
                ###                   END OF YOUR CODE                   ###
                ############################################################




