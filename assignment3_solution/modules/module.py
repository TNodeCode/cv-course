__all__ = ['Module']



class Module:

    def __init__(self):
        """
        Create base class for implementing network layers.

        Attributes:
            - param (dict): Dictionary for parameters.
            - grad (dict): Dictionary for gradients.

        """
        self.param = {}
        self.grad = {}

        # Network graph.
        self._children = []
        self._ordering = []
        self._parent = None


    def forward(self, x):
        """
        Compute forward pass through module.

        Implements the default behavior which is to apply
        the child modules in order of definition.

        Parameters:
            - x (np.array): Input features.

        Returns:
            - out: (np.array): Output features.

        """
        out = x

        for module in self._children:
            out = module(out)

        return out


    def backward(self, out_grad):
        """
        Compute backward pass through module.

        Implements the default behavior that the backward
        methods of the child modules are called in reverse
        order of the forward pass.

        Parameters:
            - out_grad (np.array): Gradient with respect to module output.

        Returns:
            - in_grad (np.array): Gradient with respect to module input.

        """
        in_grad = out_grad

        for module in reversed(self._ordering):
            in_grad = module.backward(in_grad)

        return in_grad


    def get_modules(self):
        """
        Collect all submodules of the current module.

        Also includes the module itself as the first list entry.
        The order depends on the order of definition.

        Returns:
            - modules (list[Module]): List of child modules.

        """
        modules = [self]

        for module in self._children:
            modules += module.get_modules()

        return modules


    def __getattribute__(self, name):
        """
        Define hooks for forward and backward pass.

        Updates call order in parent module if forward
        method is called and clears own call order list
        if backward pass is completed.

        """
        attribute = object.__getattribute__(self, name)

        if name == 'forward' and self._parent:

            def forward_hook(*args):
                out = attribute(*args)
                ordering = self._parent._ordering
                if self not in ordering:
                    ordering.append(self)
                return out

            return forward_hook

        if name == 'backward' and self._children:

            def backward_hook(*args):
                in_grad = attribute(*args)
                self._ordering = []
                return in_grad

            return backward_hook

        return attribute


    def __setattr__(self, name, value):
        """
        Store child modules in internal list.
        """
        is_module = value.__class__.__bases__[0] == self.__class__.__bases__[0]

        if is_module and name not in ['_parent', '__class__']:
            self._children.append(value)
            object.__setattr__(value, '_parent', self)

        object.__setattr__(self, name, value)


    def __call__(self, *args):
        """
        Make module callable such that calling an instance
        is equivalent to calling the forward method.
        """
        return self.forward(*args)


    def __str__(self):
        """
        Set module name as string representation.
        """
        return self.__class__.__name__



