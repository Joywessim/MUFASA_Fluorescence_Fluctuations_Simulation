import numpy as np

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
