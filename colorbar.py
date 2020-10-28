import cv2
import numpy as np
import matplotlib.pyplot as plt


cmap = [('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]


gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(cmap_category, cmap_list):
    # Create figure and adjust figure height to number of colormaps
    fig, ax = plt.subplots(figsize=(10,1))
    #ax.set_title(cmap_category + ' colormaps', fontsize=14)

    plt.imshow(gradient, aspect='auto', cmap=plt.get_cmap(cmap_list))
    #ax.text(-.01, .5, cmap_list, va='center', ha='right', fontsize=10,
      #    transform=ax.transAxes)


plot_color_gradients('Cyclic', 'hsv')
plt.savefig('color.png')
plt.show()
