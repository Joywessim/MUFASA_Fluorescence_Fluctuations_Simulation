import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom


def apply_blur_convolution(image, kernel_size=5, sigma=None):
    if sigma==None:
        sigma = kernel_size / 2.0
    blurred_image = gaussian_filter(image, sigma=sigma)
    return blurred_image

def undersample_image(image, factor=4):
    zoom_factor = 1 / factor
    undersampled_image = zoom(image, zoom=zoom_factor, order=0) 
    return undersampled_image
    
