import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def divide_into_frames(times, states, frame_duration, num_frames):
    frames = []

    # Calculate the start and end times for each frame
    frame_edges = np.linspace(0, num_frames * frame_duration, num_frames + 1)

    # Use digitize to assign each time to the appropriate frame
    frame_indices = np.digitize(times, frame_edges) - 1

    # Collect times and states for each frame
    for frame_idx in range(num_frames):
        in_frame = frame_indices == frame_idx
        frames.append((times[in_frame], states[in_frame]))

    return frames

def excitation_rate(epsilon, excitation_P, excitation_wavelength, N_A=6.022e23, h=6.626e-34, c=3e8):
    constant_factor = (10**-6 * np.log(10)) / (N_A * h * c)
    k01 = epsilon * excitation_P * excitation_wavelength * constant_factor    
    return k01



#image utils

def process_frame(f, fluctuations, grid_shape, adjusted_x, adjusted_y):
    grid = np.zeros(grid_shape)
    
    # Clip the adjusted x and y coordinates to ensure they are within grid bounds
    adjusted_x_sim = np.clip(adjusted_x[:fluctuations.shape[0]], 0, grid_shape[1] - 1)
    adjusted_y_sim = np.clip(adjusted_y[:fluctuations.shape[0]], 0, grid_shape[0] - 1)
    
    np.add.at(grid, (adjusted_y_sim, adjusted_x_sim), fluctuations[:, f])
    return grid


def fluctuations_to_images(fluctuations, num_frames, grid_shape, adjusted_x, adjusted_y):
    frames = Parallel(n_jobs=-1)(
        delayed(process_frame)(f, fluctuations, grid_shape, adjusted_x, adjusted_y) for f in tqdm(range(num_frames))
    )
    return frames

def prepare_adjusted_positions(adjusted_x, adjusted_y, num_fluorophores_to_simulate):
    # Select the first num_fluorophores_to_simulate positions
    adjusted_x_sim = adjusted_x[:num_fluorophores_to_simulate]
    adjusted_y_sim = adjusted_y[:num_fluorophores_to_simulate]
    return adjusted_x_sim, adjusted_y_sim

colors_fluctuations_output = ['#000000', '#F49595']    
colors_us_output = ['#000000', '#FCD5CE']    
colors_poisson_output = ['#000000', '#FEC89A']   

color_mean_poisson = "#83B8C6"
color_mean_poisson_pixel  = "#83B8C6"
color_mean_us  = '#F3D17C'
color_mean_us_pixel  = '#F3D17C'


