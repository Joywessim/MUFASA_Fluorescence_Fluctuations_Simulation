import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce
import matplotlib.pyplot as plt



import torch
import torch.nn as nn
import numpy as np

class Conv2dConstKernel(nn.Module):
    def __init__(self,     
                 kwidth,
                 ksigma,
                 undersampling,
                 dtype,
                 conv=nn.functional.conv2d) :

        super().__init__()

        self.kwidth = kwidth
        self.undersampling = undersampling
        self.conv = conv

        # build gaussian kernel
        ax = torch.linspace(-kwidth//2, kwidth//2, kwidth, dtype=dtype)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(- (xx**2 + yy**2) / 2 / ksigma**2)
        kernel /= kernel.sum()

        # reshape it to a 4D tensor
        kernel = kernel.view(1, 1, kwidth, kwidth)

        # convolve it with undersampling matrix of ones
        ones = torch.ones((1, 1, undersampling, undersampling), dtype=dtype)
        kernel = nn.functional.conv2d(kernel, ones, padding=undersampling // 2)

        # save it as a constant parameter
        self.kernel = nn.Parameter(kernel, requires_grad=False)

        # build a padding layer to keep dimensions
        self.pad = nn.ReflectionPad2d(kwidth // 2)

    def forward(self, x):
        """
        Applies convolution with constant kernel and stride to input x

        Parameters
        ----------
        x : tensor or numpy array of shape (n_images, n_channels, height, width)

        Returns
        -------
        y : tensor of same shape as x
        """
        # Check if input is a NumPy array and convert to PyTorch tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Apply padding and convolution
        return self.conv(self.pad(x), self.kernel, stride=self.undersampling)






def apply_blur_undersampling(image, kwidth=61, FWHM=318, px=103, undersampling=4):
    """
    Apply Gaussian blur and undersampling to the input image using PyTorch Conv2d.
    
    Parameters
    ----------
    image : numpy array or tensor of shape (height, width) or (1, height, width)
    kwidth : int
        Width of the Gaussian kernel.
    FWHM : float
        Full width at half maximum (for calculating the Gaussian sigma).
    px : float
        Pixel size.
    undersampling : int
        Factor by which to undersample the image.
    
    Returns
    -------
    blurred_image : torch.Tensor
        The blurred and undersampled image.
    """
    # Ensure the input is 4D (n_images, n_channels, height, width)
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = image[np.newaxis, np.newaxis, :, :]  # Add batch and channel dimensions
        elif image.ndim == 3:
            image = image[np.newaxis, :, :, :]  # Add batch dimension
        image = torch.from_numpy(image).float()  # Convert to PyTorch tensor
    
    elif isinstance(image, torch.Tensor):
        if image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elif image.ndim == 3:
            image = image.unsqueeze(0)  # Add batch dimension
    
    # Calculate Gaussian sigma
    ksigma = FWHM / 2.335 / px * undersampling
    
    # Create the Conv2dConstKernel object
    conv = Conv2dConstKernel(kwidth, ksigma, undersampling, torch.float32)
    
    # Apply the convolution (blur) to the input image
    blurred_image = conv.forward(image)
    
    return blurred_image.squeeze(0).squeeze(0)  # Remove batch and channel dimensions if needed

def apply_camera_noise(frames, QE, 
                       EM_gain, e_adu, 
                       sigma_R, c, BL):
    """
    Apply EMCCD camera noise model to the frames based on the given parameters.

    Parameters:
    - frames: The input 3D frames (num_frames, height, width).
    - QE: Quantum Efficiency (default=0.9)
    - EM_gain: Electron multiplication gain (default=300)
    - e_adu: Electron per ADU conversion factor (default=45)
    - sigma_R: Read noise in electrons (default=74.4)
    - c: Spurious charge in electrons (default=0.002)
    - BL: Baseline in ADU (default=100)
    - background: Background noise level (default=0)

    Returns:
    - noisy_frames: Frames with Poisson, shot, read, and EM noise applied.
    """
    # Apply Quantum Efficiency and spurious charge (convert photon counts to electron counts)
    frames_electrons = QE * frames + c
    
    # Shot noise (Poisson noise)
    frames_poisson = np.random.poisson(frames_electrons)
    
    # Electron multiplication noise (Gamma distribution)
    frames_em_gain = np.random.gamma(shape=frames_poisson, scale=EM_gain)
    
    print(sigma_R)
    # Read noise (Gaussian distribution) added after electron multiplication
    frames_noisy = frames_em_gain + np.random.normal(0, sigma_R, frames_em_gain.shape)
    
    # Convert to ADU (Analog to Digital Units) and apply baseline
    frames_adu = (frames_noisy-(frames_noisy % e_adu))/e_adu + BL
    # frames_adu = G * frames_noisy + BL
    
    # Clip ADU values to the maximum allowed value
    frames_adu = np.clip(frames_adu, 0, 65535)
    
    return frames_adu

def undersample(frames, factor):
    """
    Undersamples a 3D array of frames by a given factor, where the resulting pixel value 
    is the sum of the corresponding pixels in the original frames.

    Parameters:
    - frames: 3D numpy array representing the original frames (shape: num_frames, height, width).
    - factor: Integer factor by which to reduce the size of the frames.

    Returns:
    - undersampled_frames: 3D numpy array representing the undersampled frames.
    """
    # Apply block_reduce with a 3D block size: (1, factor, factor) to preserve the number of frames
    undersampled_frames = block_reduce(frames, block_size=(1, factor, factor), func=np.sum)
    return undersampled_frames

def apply_camera_advanced(frames, QE=0.9, sigma_R=74.4, c=0.002, EM_gain=300, e_adu=45, BL=100, us_factor=8,kwidth=61, FWHM = 318,px = 103, background=0):
    """
    Simulates the behavior of the EMCCD camera, applying blur, undersampling, and noise.

    Parameters:
    - frames: The input 3D frames (num_frames, height, width).
    - num_frames: Number of frames.
    - grid_shape: Shape of the input grid (height, width).
    - QE: Quantum efficiency at 700 nm absorption wavelength (default=0.9)
    - sigma_R: Read noise in electrons (default=74.4)
    - c: Spurious charge in electrons (default=0.002)
    - EM_gain: Electron multiplication gain (default=300)
    - e_adu: Electrons per ADU conversion factor (default=45)
    - BL: Baseline in ADU (default=100)
    - alpha_detection: Detection coefficient for noise generation.
    - us_factor: Undersampling factor.
    - kernel_size: Size of the blur kernel.
    - background: Background noise level (default=0).

    Returns:
    - frames_us: Undersampled frames with Poisson and Gaussian noise applied.
    """
    
    # # Determine size of undersampled images
    # us_rows = (grid_shape[0] // us_factor) if (grid_shape[0] // us_factor) % 2 == 0 else (grid_shape[0] // us_factor + 1)
    # us_cols = (grid_shape[1] // us_factor) if (grid_shape[1] // us_factor) % 2 == 0 else (grid_shape[1] // us_factor + 1)

   

    # Apply blur to all frames
    frames_blurred_us =apply_blur_undersampling(frames, kwidth=kwidth, FWHM=FWHM,px = px,undersampling=us_factor)

    

    # Apply the camera noise model
    frames_noisy = apply_camera_noise(frames_blurred_us+background, QE, EM_gain, e_adu, sigma_R, c,  BL)

    return frames_blurred_us,frames_noisy  # Return undersampled frames with noise


def apply_camera(frames, num_frames, grid_shape, alpha_detection=1, us_factor=8, sigma_noise=1, kernel_size=5, background=0):
    """
    Optimized version of apply_camera that applies blur, undersampling, and noise in-place.
    
    Parameters:
    - frames: The input 3D frames (num_frames, height, width).
    - num_frames: Number of frames.
    - grid_shape: Shape of the input grid (height, width).
    - alpha_detection: Detection coefficient for noise generation.
    - us_factor: Undersampling factor.
    - sigma_noise: Standard deviation for Gaussian noise.
    - kernel_size: Size of the blur kernel.
    
    Returns:
    - frames_us: Undersampled frames with Poisson and Gaussian noise applied.
    """
    
    # Determine size of undersampled images
    us_rows = (grid_shape[0] // us_factor) if (grid_shape[0] // us_factor) % 2 == 0 else (grid_shape[0] // us_factor + 1)
    us_cols = (grid_shape[1] // us_factor) if (grid_shape[1] // us_factor) % 2 == 0 else (grid_shape[1] // us_factor + 1)

    # Pre-allocate undersampled frames array
    frames_us = np.zeros((num_frames, us_rows, us_cols), dtype=np.float32)

    # Apply blur to all frames (vectorized)
    frames_blurred = apply_blur(frames, kernel_size )

    # Undersample all blurred frames (vectorized)
    frames_us = undersample(frames_blurred, us_factor) 

    # Apply Poisson noise and Gaussian noise in-place
    frames_noisy = alpha_detection * np.random.poisson(frames_us + background)  # In-place Poisson noise
    frames_noisy= frames_noisy + np.random.normal(scale=sigma_noise, size=frames_us.shape)  # In-place Gaussian noise

    return frames_us,frames_noisy  # Return undersampled frames with noise





def gaussian_noise(sigma): 
    return np.random.normal(scale=sigma)


# Test the camera model on dummy frames
if __name__ == "__main__":
    # Example usage
    num_frames = 100
    grid_shape = (1024, 1024)
    QE=0.9
    sigma_R=0
    c=0.002
    EM_gain=300
    e_adu=45
    BL=100
    us_factor=8
    kernel_size=5
    
    # Load example frames or use simulated data
    frames = np.load("C:/Users/omezz/ISSS/frames.npy")[0][np.newaxis,:] # Simulated photon counts
    
    # Apply the camera simulation
    frames_us,noisy_frames = apply_camera_simulation(frames, 
                                           num_frames, 
                                           grid_shape, 
                                           QE=0.9, 
                                           sigma_R=0, 
                                           c=0.002, 
                                           EM_gain=300, 
                                           e_adu=45, 
                                           BL=100, 
                                           us_factor=8, 
                                           kernel_size=5,
                                           background=0)

    # Visualize the first frame with noise applied
    plt.figure()
    plt.imshow(noisy_frames[0], cmap='gray')
    plt.title("Simulated Frame with Camera Noise")
    plt.colorbar()

    plt.figure()
    plt.imshow(frames_us[0], cmap='gray')
    plt.title("Simulated Frame with Camera Noise")
    plt.colorbar()
    plt.show()



    print("Processed frames with noise.")
