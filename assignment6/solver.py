from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.notebook import tnrange
from tqdm import tqdm

from metrics import stat_scores, iou_score, accuracy_score


class Solver:

    def __init__(self, train_dataloader, val_dataloader, model, loss, device):
        """
        Creates a solver for semantic segmentation.
        """
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Create loss function.
        self.loss = loss

        # Create optimizer.
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

        # Scheduler is optional.
        self.scheduler = None

        # Some attributes for bookkeeping.
        self.epoch = 0
        self.num_epochs = 0
        self.history = defaultdict(list)

    @torch.no_grad()
    def infer(self, inputs):
        """
        Predicts the mask for a batch of images by forwarding the inputs through the model,
        converts the logits to a probability and returns the class with the
        highest probability.

        Parameters:
            - inputs (torch.Tensor): Batch of images. Shape: (N, 3, H, W)

        Returns:
            - preds (torch.Tensor): Predicted masks. Shape: (N, H, W)
        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################
        preds = None

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return preds

    def test(self, dataloader):
        """
        Compute the iou and accuracy of the model.

        Parameters:
            - dataloader (torch.utils.data.DataLoader): Dataloader for the dataset.

        Returns:
            - iou (float): Intersection over union score.
            - accuracy (float): Percentage of correct predictions.

        """
        self.model.to(self.device)
        self.model.eval()
        epoch_iou, epoch_accuracy = [], []
        with torch.no_grad():
            for inputs, masks in dataloader:
                # Transfer data to selected device.
                inputs = inputs.to(self.device)  # (N, 3, H, W)
                masks = masks.to(self.device)  # (N, H, W)

                preds = self.infer(inputs)  # (N, H, W)

                ############################################################
                ###                  START OF YOUR CODE                  ###
                ############################################################
                # fill epoch_iou and epoch_accuracy with the iou and accuracy

                ############################################################
                ###                   END OF YOUR CODE                   ###
                ############################################################
        self.model.train()

        return np.mean(epoch_iou), np.mean(epoch_accuracy)

    def train(self, num_epochs=10):
        """
        Train the model for given number of epochs.

        Parameters:
            - num_epochs (int): Number of epochs to train.

        Returns:
            - history (dict):
                - loss: Training set loss per epoch.
                - train_acc: Training set accuracy per epoch.
                - train_iou: Training set iou per epoch.
                - val_acc: Validation set accuracy per epoch.
                - val_iou: Validation set iou per epoch.

        """
        self.model.to(self.device)
        self.num_epochs += num_epochs

        # Keep track of best iou and model parameters.
        best_val_iou = 0
        best_params = None

        for epoch in (pbar := tqdm(range(num_epochs), unit='epoch')):
            self.epoch += 1
            loss_history = []

            for i, (inputs, masks) in enumerate(self.train_dataloader):
                # Transfer inputs and labels to selected device.
                inputs = inputs.to(self.device)  # (N, 3, H, W)
                masks = masks.to(self.device)  # (N, H, W)

                # Compute forward pass through network.
                logits = self.model(inputs)  # (N, C, H, W)

                ############################################################
                ###                  START OF YOUR CODE                  ###
                ############################################################
                # create one-hot encoding for masks
                target = None

                ############################################################
                ###                   END OF YOUR CODE                   ###
                ############################################################

                # Compute loss and gradients for model parameters.
                loss = self.loss(logits, target)
                loss.backward()

                # Update model parameters and reset gradients.
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Store current loss.
                loss_history.append(loss.item())

            # Store average loss per epoch.
            self.history['loss'].append(sum(loss_history) / i)

            # Check and store training accuracy.
            train_iou, train_acc = self.test(self.train_dataloader)
            self.history['train_iou'].append(train_iou)
            self.history['train_acc'].append(train_acc)

            # Check and store validation accuracy.
            val_iou, val_acc = self.test(self.val_dataloader)
            self.history['val_iou'].append(val_iou)
            self.history['val_acc'].append(val_acc)

            # Update learning rate.
            if self.scheduler:
                self.scheduler.step()

            # Update best accuracy and model parameters.
            if val_iou > best_val_iou:
                best_params = self.model.state_dict().copy()
                best_val_iou = val_iou

            # Show current validation set accuracy.
            pbar.set_description(f'Validation iou: {val_iou:5.2f} Validation accuracy: {val_acc:5.2f}')

        # Swap best parameters from training into the model.
        self.model.load_state_dict(best_params)

        return self.history
