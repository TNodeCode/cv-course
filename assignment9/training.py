import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import autograd
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm



__all__ = ['Game']



def iterate(loader, num_iter):
    """
    Get arbitrary amount of samples from data loader.

    Parameters:
        - loader (torch.utils.data.DataLoader): Data loader.
        - num_iter (int): Number of iterations.

    Yields:
        - Data loader output.

    """
    iterator = iter(loader)

    # Iterate data loader until exhaustion.
    for _ in range(num_iter):
        try:
            yield next(iterator)

        # Create new iterator and continue.
        except StopIteration:
            iterator = iter(loader)

            yield next(iterator)



def requires_grad(model, flag):
    """
    Set whether model parameters require gradient computation.

    Parameters:
        - flag (bool): Gradient computation required.

    """
    ############################################################
    ###                  START OF YOUR CODE                  ###
    ############################################################

    for param in model.features.parameters():
        param.requires_grad = flag

    ############################################################
    ###                   END OF YOUR CODE                   ###
    ############################################################



def discriminator_loss(real_pred, fake_pred):
    """
    Compute loss for discriminator network.

    Parameters:
        - real_pred (torch.Tensor): Predictions for real images.
        - fake_pred (torch.Tensor): Predictions for fake images.

    Returns:
        - loss (torch.Tensor): Discriminator loss.

    """
    ############################################################
    ###                  START OF YOUR CODE                  ###
    ############################################################

    loss = -0.5 * (torch.log(real_pred) + torch.log(1 - fake_pred)).mean()

    ############################################################
    ###                   END OF YOUR CODE                   ###
    ############################################################
    return loss



def generator_loss(fake_pred):
    """
    Compute loss for generator network.

    Parameters:
        - fake_pred (torch.Tensor): Predictions for fake images.

    Returns:
        - loss (torch.Tensor): Generator loss.

    """
    ############################################################
    ###                  START OF YOUR CODE                  ###
    ############################################################

    loss = -0.5 * torch.log(fake_pred).mean()

    ############################################################
    ###                   END OF YOUR CODE                   ###
    ############################################################
    return loss



class Game:

    def __init__(self, discriminator, generator, dataloader, device, **kwargs):
        """
        Create generative adversarial network game.

        Parameters:
            - discriminator (nn.Module): Discriminator network.
            - generator (nn.Module): Generator network.
            - dataloader (DataLoader): Loader for dataset.
            - device (torch.Device): Device for training.
            - lr (float) [0.0002]: Learning rate.
            - batch_size [128] (int): Number of samples per batch.
            - latent_dim [100] (int): Size of latent vectors.
            - show_every (int) [1000]: Interval for showing images.
            - num_images (int) [100]: Number of images to show.
            - prefix (str) ['test']: Prefix for saving files.

        """
        self.d = discriminator
        self.g = generator
        self.dataloader = dataloader
        self.device = device

        # Set default values.
        defaults = {
            'lr': 0.0002,
            'batch_size': 128,
            'latent_dim': 100,
            'show_every': 1_000,
            'num_images': 100,
            'prefix': 'test'
        }

        # Get given argument or take default value.
        values = defaults | kwargs

        # Get learning rate.
        lr = values.pop('lr')

        # Create both optimizers.
        self.d_optim = torch.optim.Adam(self.d.parameters(), lr=lr, betas=(0.5, 0.999))
        self.g_optim = torch.optim.Adam(self.g.parameters(), lr=lr, betas=(0.5, 0.999))

        # Store remaining arguments.
        self.__dict__ |= values

        # Setup bookkeeping.
        self.d_loss = []
        self.g_loss = []
        self.iter = 0


    def save(self, save_dir):
        """
        Save networks and training state.

        Parameters:
            - save_dir (str): Directory to save state.

        """
        state = {

            # Save network parameters.
            'd': self.d.state_dict(),
            'g': self.g.state_dict(),

            # Save optimizer state.
            'd_optim': self.d_optim.state_dict(),
            'g_optim': self.g_optim.state_dict(),

            # Save losses.
            'd_loss': self.d_loss,
            'g_loss': self.g_loss,

            # Save number of iterations.
            'iter': self.iter
        }

        torch.save(state, f'{save_dir}/{self.prefix}_{str(self.iter).zfill(6)}.pt')


    def load(self, checkpoint):
        """
        Load networks and training state from disk.

        Parameters:
            - checkpoint (str): Filename of stored state.

        """
        self.d.to(self.device)
        self.g.to(self.device)

        state = torch.load(checkpoint)

        # Load trained parameters into models.
        self.d.load_state_dict(state['d'])
        self.g.load_state_dict(state['g'])

        # Load optimizer states.
        self.d_optim.load_state_dict(state['d_optim'])
        self.g_optim.load_state_dict(state['g_optim'])

        # Load losses.
        self.d_loss = state['d_loss']
        self.g_loss = state['g_loss']

        # Load iterations.
        self.iter = state['iter']


    def show(self, save_dir=None):
        """
        Generate image grid to show progress.

        Parameters:
            - save_dir (str|None, optional): Directory for storing images.

        """
        self.g.to(self.device)

        with torch.no_grad():
            self.g.eval()

            # Create noise vectors and generate images.
            noise = torch.randn(self.num_images, self.latent_dim).to(self.device)
            outputs = self.g(noise)

            # Align output images in grid.
            image = make_grid(
                tensor=outputs,
                nrow=int(self.num_images**0.5),
                normalize=True,
                value_range=(-1, 1)
            )

            # Normalize and store images.
            if save_dir is not None:
                save_image(image, f'{save_dir}/{self.prefix}_{str(self.iter).zfill(6)}.jpg')

            self.g.train()

        return image.cpu()


    def play(self, num_iter):
        """
        Train networks for given number of iterations.

        Parameters:
            - num_iter (int): Number of iterations for training run.

        Returns:
            - (dict): Training history dictionary.

        """
        self.d.to(self.device)
        self.g.to(self.device)

        # Set format for progress bar.
        bar_format = '{l_bar}{bar:20}{r_bar}{bar:-10b}'

        for inputs, targets in tqdm(iterate(self.dataloader, num_iter), bar_format=bar_format, total=num_iter):
            self.iter += 1
            ############################################################
            ###                  START OF YOUR CODE                  ###
            ############################################################

            batch_size, channels, img_height, img_width = inputs.shape
            
            ### TRAIN DESCRIMINATOR ###
            
            # Reset gradients of discriminator
            self.d.zero_grad()
            
            # Forward pass real batch through D
            pred_real = self.d(inputs)
            
            # Create noise vector that is fed into the generator
            noise = torch.randn((batch_size, 100, 1, 1)).to(self.device)
            
            # Generate fake image batch with generator
            fake_images = self.g(noise)
            
            # Classify all fake batch with discriminator
            pred_fake = self.d(fake_images.detach()).view(-1)
            
            # Calculate discriminator's loss on the real and fake images
            loss_d = discriminator_loss(pred_real, pred_fake)
            
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            loss_d.backward()
            
            # Update weights of descriminator
            self.d_optim.step()
            
            ### TRAIN GENERATOR ###
            
            # Reset gradients of generator
            self.g.zero_grad()
            
            # Run fake images through discriminator
            pred_fake = self.d(fake_images).view(-1)
            
            # Compute generator loss
            loss_g = generator_loss(pred_fake)
            
            # Compute gradients for generator
            loss_g.backward()
            
            # Update weights of generator
            self.g_optim.step()
            
            if not self.iter % 1000:
                print("D_LOSS", loss_d.item(), "G_LOSS", loss_g.item())
                
            
            ############################################################
            ###                   END OF YOUR CODE                   ###
            ############################################################

            # Save image grid for visual inspection.
            if self.iter % self.show_every == 0:
                self.show('./images')

        return {
            'd_loss': self.d_loss,
            'g_loss': self.g_loss,
            'iter': self.iter
        }





