import matplotlib.pyplot as plt



def show_training(history):
    """
    Show loss and accuracies during training.

    Parameters:
        - history (dict):
            - loss (list[float]): Training losses.
            - train_acc (list[float]): Training accuracies.
            - val_acc (list[float]): Validation accuracies.

    """
    fig, (lhs, rhs) = plt.subplots(ncols=2, figsize=(12, 4))
    fig.suptitle('Training')

    # Set subplot titles.
    lhs.set_title('Loss')
    rhs.set_title('Accuracy')

    # Set subplot axis labels.
    lhs.set_xlabel('epoch'), lhs.set_ylabel('loss')
    rhs.set_xlabel('epoch'), rhs.set_ylabel('accuracy')

    # Plot loss and accuracies.
    lhs.plot(history['loss'])
    rhs.plot(history['train_acc'], label='train')
    rhs.plot(history['val_acc'], label='val')
    rhs.legend()

    plt.show()



