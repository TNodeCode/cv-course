import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from tqdm.notebook import tnrange


class Solver:

    def __init__(self, model, data, **kwargs):
        """
        Creates a solver for classification.

        Parameters:
            - model (nn.Module):
                  Model to be trained.
            - data (dict):
                  Training and validation datasets.
                  Dictionary with keys `train` for training set and `val` for validation set.
            - loss (str):
                  Class name of the loss function to be optimized.
                  [Default: 'CrossEntropyLoss']
            - loss_config (dict|None):
                  Dictionary with keyword arguments for calling the loss function.
                  [Default: {}]
            - optimizer (str):
                  Class name of the optimizer to be used.
                  [Default: 'SGD']
            - optimizer_config (dict):
                  Dictionary with keyword arguments for calling for the optimizer.
                  Model parameters don't have to be passed explicitly.
                  [Default: {'lr': 1e-2}]
            - batch_size (int):
                  Number of samples per minibatch.
                  [Default: 128]
            - num_train_samples (int):
                  Number of training samples to be used for evaluation.
                  [Default: 1000]
            - num_val_samples (int|None):
                  Number of validation samples to be used for evaluation.
                  If parameter is `None`, all samples in the given validation set are used.
                  [Default: None]
            - scheduler (str|None):
                  Class name of the learning rate scheduler to be used.
                  If parameter is not given or `None`, no scheduler is used.
                  [Default: None]
            - scheduler_config (dict):
                  Dictionary with keyword arguments to provide for the scheduler.
                  The optimizer is passed in automatically.
                  [Default: {}]
            - metric (str):
                  Metric to be used for measure performance. Torchmetrics class.
                  [Default: 'Accuracy']
            - metric_config (dict):
                  Dictionary with keyword arguments for calling the metric.
                  [Default: {}]
        """
        self.model = model

        # Train on the GPU if possible.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Store training and validation data.
        self.data_train = data['train']
        self.data_val = data['val']

        # Define default values for parameters.
        defaults = {
            'loss': 'CrossEntropyLoss',
            'loss_config': {},
            'optimizer': 'SGD',
            'optimizer_params': {'lr': 1e-2},
            'batch_size': 128,
            'num_train_samples': 1000,
            'num_val_samples': None,
            'scheduler': None,
            'scheduler_config': {}
        }

        # Get given argument or take default value.
        values = defaults | kwargs

        # Create loss function.
        loss = getattr(nn, values.pop('loss'))
        self.loss = loss(**values.pop('loss_config'))

        # Create optimizer.
        optimizer = getattr(torch.optim, values.pop('optimizer'))
        self.optimizer = optimizer(model.parameters(), **values.pop('optimizer_config'))

        # Scheduler is optional.
        self.scheduler = values.pop('scheduler')

        # Create scheduler if necessary.
        if self.scheduler:
            scheduler = getattr(torch.optim.lr_scheduler, self.scheduler)
            self.scheduler = scheduler(self.optimizer, **values.pop('scheduler_config'))

        # Create metric.
        metric = getattr(torchmetrics, values.pop('metric'))
        self.metric = metric(**values.pop('metric_config'))

        # Store remaining arguments.
        self.__dict__ |= values

        # Some attributes for bookkeeping.
        self.epoch = 0
        self.num_epochs = 0
        self.loss_history = []
        self.train_acc = []
        self.val_acc = []

    def save(self, path):
        """
        Save model and training state to disk.

        Parameters:
            - path (str): Path to store checkpoint.

        """
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'num_epochs': self.num_epochs,
            'loss_history': self.loss_history,
            'train_acc': self.train_acc,
            'val_acc': self.val_acc
        }

        # Save learning rate scheduler state if defined.
        if self.scheduler:
            checkpoint['scheduler'] = self.scheduler.state_dict()

        # Save checkpoint to disk.
        torch.save(checkpoint, path)

    def load(self, path):
        """
        Load checkpoint from disk.

        Parameters:
            - path (str): Path to checkpoint.

        """
        checkpoint = torch.load(path)

        # Load model and optimizer state.
        self.model.load_state_dict(checkpoint.pop('model'))
        self.optimizer.load_state_dict(checkpoint.pop('optimizer'))

        # Load learning rate scheduler state if defined.
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint.pop('scheduler'))

        # Load the remaining attributes.
        self.__dict__ |= checkpoint

    def test(self, dataset, num_samples=None):
        """
        Compute the accuracy of the model.

        Takes an optional parameter that allows to specify the
        number of samples to use for testing. If not given the
        whole dataset is used.

        Parameters:
            - dataset (torch.Tensor): Dataset for testing.
            - num_samples (int|None): Number of data points to use from dataset.

        Returns:
            - accuracy (float): Percentage of correct predictions.

        """
        self.model.to(self.device)
        self.model.eval()
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        # Get number of samples in dataset.
        dataset_size = len(dataset)

        # Subsample data when needed.
        if num_samples and num_samples < dataset_size:
            dataset, _ = torch.utils.data.random_split(dataset, [num_samples, dataset_size - num_samples])

        # Create loader for dataset.
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

        with torch.no_grad():
            for inputs, labels in data_loader:
                # Transfer data to selected device.
                if isinstance(inputs, (list, tuple)):
                    inputs = [input.to(self.device).to(torch.float32) for input in inputs if isinstance(input, torch.Tensor)]
                else:
                    inputs = inputs.to(self.device).to(torch.float32)
                labels = labels.to(self.device).to(torch.int64)

                # Compute forward pass.
                outputs = self.model(inputs)

                self.metric(outputs, labels)

        # Accumulate metric.
        result = self.metric.compute()

        # Reset metric.
        self.metric.reset()
        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        self.model.train()

        return result.cpu().item()

    def train(self, num_epochs=10):
        """
        Train the model for given number of epochs.

        Parameters:
            - num_epochs (int): Number of epochs to train.

        Returns:
            - history (dict):
                - loss: Training set loss per epoch.
                - train_acc: Training set accuracy per epoch.
                - val_acc: Validation set accuracy per epoch.

        """
        self.model.to(self.device)
        self.num_epochs += num_epochs
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        # Keep track of best accuracy and model parameters.
        best_val_acc = 0
        best_params = None

        # Create data loader for training set.
        train_loader = DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

        for epoch in (pbar := tnrange(num_epochs)):
            self.epoch += 1
            loss_history = []

            for i, (inputs, labels) in enumerate(train_loader):
                # Transfer inputs and labels to selected device.
                if isinstance(inputs, (list, tuple)):
                    inputs = [input.to(self.device) for input in inputs if isinstance(input, torch.Tensor)]
                else:
                    inputs = inputs.to(self.device)

                labels = labels.to(self.device)

                # Compute forward pass through network.
                outputs = self.model(inputs)

                # Compute loss and gradients for model parameters.
                loss = self.loss(outputs, labels)
                loss.backward()

                # Update model parameters and reset gradients.
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Store current loss.
                loss_history.append(loss.item())

            # Store average loss per epoch.
            self.loss_history.append(sum(loss_history) / i)

            # Check and store training accuracy.
            train_acc = self.test(self.data_train, self.num_train_samples)
            self.train_acc.append(train_acc)

            # Check and store validation accuracy.
            val_acc = self.test(self.data_val, self.num_val_samples)
            self.val_acc.append(val_acc)

            # Update learning rate.
            if self.scheduler:
                self.scheduler.step()

            # Update best accuracy and model parameters.
            if val_acc > best_val_acc:
                best_params = self.model.state_dict().copy()
                best_val_acc = val_acc

            # Show current validation set accuracy.
            pbar.set_description(f'Validation accuracy: {val_acc:5.2f}%')

        # Swap best parameters from training into the model.
        self.model.load_state_dict(best_params)

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return {
            'loss': self.loss_history,
            'train_acc': self.train_acc,
            'val_acc': self.val_acc
        }


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
    rhs.set_title('mAP')

    # Set subplot axis labels.
    lhs.set_xlabel('epoch'), lhs.set_ylabel('loss')
    rhs.set_xlabel('epoch'), rhs.set_ylabel('mAP')

    # Plot loss and accuracies.
    lhs.plot(history['loss'])
    rhs.plot(history['train_acc'], label='train')
    rhs.plot(history['val_acc'], label='val')
    rhs.legend()

    plt.show()
