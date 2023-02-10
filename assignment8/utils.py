import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt



def num_params(module):
    """
    Compute number of parameters.

    Parameters:
        - module (nn.Module): Module with parameters.

    Returns:
        - (int): Number of learnable parameters.

    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)



def show_images(loader):
    """
    Show image grid.

    Parameters:
        - loader (torch.utils.data.DataLoader): Loader to take images from.

    """
    plt.figure(figsize=(5, 5))
    plt.axis(False)

    # Get some random samples.
    images, _ = next(iter(loader))

    # Make image grid.
    images = torchvision.utils.make_grid(images).numpy().transpose(1, 2, 0)

    # Show samples.
    plt.imshow(images)



def check_accuracy(model, loader):
    """
    Get the predictive accuracy of a model for the given dataset.
    
    Parameters:
        - model (nn.Module): The model to evaluate.
        - loader (torch.utils.data.DataLoader): Loader for dataset.

    Returns:
        - accuracy (float): Fraction of correct predictions.

    """
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in loader:

            # Reshape inputs.
            inputs = inputs.view(-1, 28, 28)

            # Compute forward pass.
            outputs = model(inputs)

            # Compute model predictions.
            predicted = torch.argmax(outputs.data, 1)

            # Increment counters.
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Compute accuracy.
    accuracy = 100*correct / total

    return accuracy



def train(model, loss, optimizer, loader_train, loader_val, num_epochs=10, print_every=1, verbose=True):
    """
    Trains a model for a specified number of epochs.

    Parameters:
        - model (nn.Module): Model to train.
        - loss (nn.Module): Loss function to use for training.
        - optimizer (nn.Module): Optimizer to use for training.
        - loader_train (torch.utils.data.DataLoader): Dataloader for training set.
        - loader_val (torch.utils.data.DataLoader): Dataloader for validation set.
        - num_epochs (int): Number of epochs.
        - print_every (int): Interval for printing results.
        - verbose (bool): Print intermediate results.

    Returns:
        - dictionary with keys:
            - loss (list[float]): Recorded loss history.
            - train_acc (list[float]): Recorded training accuracy.
            - val_acc (list[float]): Recorded validation accuracy.

    """
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):

        # Enable training mode.
        model.train()

        for inputs, labels in loader_train:

            # Reshape inputs.
            inputs = inputs.view(-1, 28, 28)

            # Compute forward pass through network.
            outputs = model(inputs)

            # Compute loss and gradients for model parameters.
            current_loss = loss(outputs, labels)
            current_loss.backward()

            # Update model parameters and reset gradients.
            optimizer.step()
            optimizer.zero_grad()

            # Store current loss.
            loss_history.append(current_loss.item())

        # Enable evaluation mode.
        model.eval()

        # Check and store training accuracy.
        train_acc = check_accuracy(model, loader_train)
        train_acc_history.append(train_acc)

        # Check and store validation accuracy.
        val_acc = check_accuracy(model, loader_val)
        val_acc_history.append(val_acc)

        # Display loss and accuracy.
        if verbose and epoch % print_every == 0:
            print(
                f'Epoch: {epoch+1:3}/{num_epochs} ',
                f'Loss: {loss_history[-1]:.5f} ',
                f'Train accuracy: {train_acc:.2f} % ',
                f'Val accuracy: {val_acc:.2f} %'
            )

    # Return training history.
    return {
        'loss': loss_history,
        'train_acc': train_acc_history,
        'val_acc': val_acc_history
    }

