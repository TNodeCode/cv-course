import matplotlib.pyplot as plt


def plot_segmentation(image, mask, pred=None, save=None):
    fig, ax = plt.subplots(1, 2 if pred is None else 3, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].axis('off')
    ax[0].set_title('Image')
    ax[1].imshow(mask)
    ax[1].axis('off')
    ax[1].set_title('Mask')

    if pred is not None:
        ax[2].imshow(pred)
        ax[2].axis('off')
        ax[2].set_title('Prediction')

    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    plt.show()


def plot_history(history, save=None):
    fig, (lhs, mid, rhs) = plt.subplots(ncols=3, figsize=(14, 6))
    fig.suptitle('Training')

    # Set subplot titles.
    lhs.set_title('Loss')
    mid.set_title('IoU')
    rhs.set_title('Accuracy')

    # Set subplot axis labels.
    lhs.set_xlabel('epoch'), lhs.set_ylabel('loss')
    mid.set_xlabel('epoch'), mid.set_ylabel('iou')
    rhs.set_xlabel('epoch'), rhs.set_ylabel('accuracy')

    # Plot loss and accuracies.
    lhs.plot(history['loss'])
    mid.plot(history['train_iou'], label='train')
    mid.plot(history['val_iou'], label='val')
    rhs.plot(history['train_acc'], label='train')
    rhs.plot(history['val_acc'], label='val')
    rhs.legend()

    if save is not None:
        plt.savefig(save)

    plt.show()
