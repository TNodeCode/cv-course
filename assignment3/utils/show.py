import numpy as np
import matplotlib.pyplot as plt



__all__ = ['show_activation', 'show_training']



def show_activation(act, val_range):
    """
    Make line plots of the activation and its derivative.

    Parameters:
        - act (Module): Activation function.
        - val_range (tuple[float|int]): Range for input values.

    """
    num = 100

    # Create figure with two subplots and title.
    fig, (lhs, rhs) = plt.subplots(ncols=2, figsize=(10, 4))
    fig.suptitle(str(act))

    # Set subplot titles.
    lhs.set_title('forward')
    rhs.set_title('backward')

    # Set subplot axis labels.
    lhs.set_xlabel('x'), lhs.set_ylabel('y')
    rhs.set_xlabel('x'), rhs.set_ylabel('dx')

    # Generate inputs and apply function.
    x = np.linspace(*val_range, num)
    y = act.forward(x)

    # Compute derivatives.
    dx = act.backward(np.ones(num))

    # Create lines along zero.
    lhs.plot(x, np.full(num, 0), alpha=0.5, c='gray')
    rhs.plot(x, np.full(num, 0), alpha=0.5, c='gray')

    # Make line plots.
    lhs.plot(x, y)
    rhs.plot(x, dx)

    plt.show()



def show_training(train_loss, val_loss, train_acc, val_acc):
    """
    Show losses and accuracies during training.

    Parameters:
        - train_loss (list[float]): Training losses.
        - val_loss (list[float]): Validation losses.
        - train_acc (list[float]): Training accuracies.
        - val_acc (list[float]): Validation accuracies.

    """
    fig, (lhs, rhs) = plt.subplots(ncols=2, figsize=(12, 4))
    fig.suptitle('Training')

    # Set subplot titles.
    lhs.set_title('Loss')
    rhs.set_title('Accuracy')

    # Set subplot axis labels.
    lhs.set_xlabel('iteration'), lhs.set_ylabel('loss')
    rhs.set_xlabel('iteration'), rhs.set_ylabel('accuracy')

    # Plot losses.
    lhs.plot(train_loss, label='train')
    lhs.plot(val_loss, label='val')
    lhs.legend()

    # Plot accuracies.
    rhs.plot(np.array(train_acc)*100, label='train')
    rhs.plot(np.array(val_acc)*100, label='val')
    rhs.legend()

    plt.show()


