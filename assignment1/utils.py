import numpy as np
import matplotlib.pyplot as plt



def show_image(image):
    """
    Callback to show an image.

    Parameters:
        - image: Image to show.

    """
    plt.imshow(image)



def show_histogram(hist, bins=256):
    """
    Callback to plot histograms.

    Parameters:
        - hist: Histogram to show.
        - bins: Number of bins.

    """
    plt.hist(np.ravel(hist), bins, [0, bins-1])



def show_plot(data):
    """
    Callback to plot data.

    Parameters:
        - data: Data to plot.

    """
    plt.plot(data)



def display(figsize, data):
    """
    Show one or more images or plots in a single figure.

    Parameters:
        - figsize: Size of the figure as tuple with (width, height).
        - data: List of tuples containing:
            - Input to show.
            - Title for subplot.
            - Position in figure.
            - Boolean to show axis or not.
            - Function to show input.

    Notes:
        - Position can be given as integer with three digits:

              <position> := <number-of-rows><number-of-columns><column-index>

        - Boolean for showing axis is optional, if no callback is given.
          Default is `False`.
        - Function to show input is optional.
          Default is `show_image`.

    """
    plt.figure(figsize=figsize)

    # Set default values.
    axis = False
    show = show_image

    for params in data:

        # Unpack parameters for current input.
        if   len(params) == 3: data, title, pos = params
        elif len(params) == 4: data, title, pos, axis = params
        elif len(params) == 5: data, title, pos, axis, show = params

        # Create titled subplot at given position.
        plt.subplot(*pos if isinstance(pos, tuple) else [pos])
        plt.title(title)

        # Enable or disable axis.
        plt.axis(axis)

        # Render input into current subplot.
        show(data)

    # Display the figure.
    plt.show()


