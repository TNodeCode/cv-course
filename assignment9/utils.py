import torch
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns



__all__ = ['count_params', 'show_loss']



# Set device for computations.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Set default theme.
sns.set_theme()



def count_params(module):
    """
    Count number of parameters in given module.

    Parameters:
        - module (nn.Module): Layer or network with parameters.

    Returns:
        - (int): Number of parameters.

    """
    return sum(param.numel() for param in module.parameters())



def moving_average(x, kernel_size=255):
    """
    Compute a moving average over the given sequence using convolution.

    Parameters:
        - x (torch.tensor):
              Sequence given as one dimensional array to compute moving average for.

        - kernel_size (int, optional) [default: 255]:
              Odd size of the convolution kernel to use for smoothing the input.

    Returns:
        - avg (torch.tensor):
              Moving average with same length as the input.

    """
    kernel = torch.ones((1, 1, kernel_size)).to(device) / kernel_size

    # Padding for equal length.
    pad = kernel_size // 2

    # Have to introduce batch and channel dimensions.
    x = torch.nn.functional.pad(x.to(device).view(1, 1, -1), (pad, pad), mode='replicate')

    # Convolve input and reshape.
    avg = torch.nn.functional.conv1d(x, kernel, padding='valid').view(-1)

    return avg.cpu()



def show_loss(history, save_dir='./images', file_name=None):
    """
    Plot losses from GAN training.

    Optionally saves the created figure in save path with given file name.
    Figure will be saved if a file name is given.

    Parameters:
        - history (dict[str, list[float]]):
              Training history for each iteration stored in a dictionary with entries:
                  - 'd_loss': Losses of the discriminator.
                  - 'g_loss': Losses of the generator.

        - save_dir (str, optional) ['./images']:
              Directory where figure is saved.

        - file_name (str|None, optional) [None]:
              File name without extension to use when saving the figure.

    """
    fig = plt.figure(figsize=(8, 5))

    # Set title and label axis.
    plt.title('Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    d_loss = history['d_loss']
    g_loss = history['g_loss']

    plt.plot(d_loss, alpha=0.2, label='D loss', c='steelblue')
    plt.plot(g_loss, alpha=0.2, label='G loss', c='skyblue')

    # Plot moving averages.
    d_loss_avg = moving_average(torch.tensor(d_loss))
    g_loss_avg = moving_average(torch.tensor(g_loss))

    plt.plot(d_loss_avg, c='steelblue')
    plt.plot(g_loss_avg, c='skyblue')

    plt.legend()

    if file_name:
        plt.savefig(f'{save_dir}/{file_name}.jpg')

    plt.show()




