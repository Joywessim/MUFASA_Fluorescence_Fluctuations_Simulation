import numpy as np

from utils import (divide_into_frames,
                   excitation_rate,
                   )

def  simulate_ctmc_one_molecule(Q, initial_state, experiment_duration,protocol, seed=None):
    
    
    if seed is not None:
        np.random.seed(seed)

    n_states = Q.shape[0]
    current_time = 0.0
    current_state = initial_state

    times = []
    states = []
    
    

    while current_time < experiment_duration:
        rate = -Q[current_state, current_state]
        if rate == 0:
            break

        # Sample the time to the next transition
        time_to_next = np.random.exponential(1.0 / rate)
        next_time = current_time + time_to_next

        if next_time > experiment_duration:
            break

        # Transition to the next state
        transition_probs = Q[current_state].copy()
        transition_probs[current_state] = 0
        next_state = np.random.choice(n_states, p=transition_probs / transition_probs.sum())

        times.append(next_time)
        states.append(next_state)

        # Update for next iteration
        current_time = next_time
        current_state = next_state
    if len(states) == 0:
        return np.array([0.0] + times), np.array([initial_state] + states , dtype=np.int8) - 1
    else:
        if protocol == "PALM":
            return np.array([0.0] + times + [experiment_duration]), np.array([initial_state] + states + [states[-1]], dtype=np.int8) - 1
        else:
            return np.array([0.0] + times + [experiment_duration]), np.array([initial_state] + states + [states[-1]], dtype=np.int8)


def count_photon_cycles_in_frames(times,states,frame_duration, num_frames, photon_emitting_state=1, ground_state=0):

    frames =  divide_into_frames(times, states, frame_duration, num_frames)
    photon_cycles = np.zeros(len(frames), dtype=np.int32)

    for i, (_, states) in enumerate(frames):
        # Find the indices where the state is photon_emitting_state
        emission_indices = np.where(states[:-1] == photon_emitting_state)[0]
        # Check if the next state is the ground_state
        photon_cycles[i] = np.sum(states[emission_indices + 1] == ground_state)

    return photon_cycles


def rate_matrix(protocol, epsilon, excitation_lifetime, excitation_wavelength,N_c, excitation_P, alpha_nr, d_E, activation_P,  alpha_isc = 1e-4):
    if protocol == "Fluctuations":
        initial_state = 0
        k_01, k_10, k_1T, k_T0, k_1B = compute_rates(epsilon, excitation_lifetime, excitation_wavelength,N_c, excitation_P , alpha_nr, d_E,alpha_isc, activation_P=None)
        return np.array(
            [[-k_01, k_01,0],
             [k_10, -(k_10 + k_1B), k_1B], 
             [0,0,0]]
            ), initial_state
    elif protocol == "STORM" or protocol == "Blinking":
        print("why am i here ?", protocol)

        initial_state = 2
        k_01, k_10, k_1T, k_T0, k_1B = compute_rates(epsilon, excitation_lifetime, excitation_wavelength,N_c, excitation_P , alpha_nr, d_E,alpha_isc, activation_P=None)
        return np.array(
            [[-k_01, k_01,0,0],
             [k_10, -(k_10 + k_1B + k_1T), k_1T, k_1B], 
             [k_T0, 0, -k_T0, 0], 
             [0,0,0,0]]
            ),initial_state
    
    elif protocol == "PALM":
        
        initial_state = 0
        k_na, k_01, k_10, k_1T, k_T0, k_1B = compute_rates(epsilon, excitation_lifetime, excitation_wavelength,N_c, excitation_P , alpha_nr, d_E,alpha_isc, activation_P)
        return np.array(
            [[-k_na,k_na, 0,0],
             [0,-k_01, k_01,0],
             [0,k_10, -(k_10 + k_1B), k_1B], 
             [0,0,0,0]]
            ), initial_state





def compute_rates(epsilon, excitation_lifetime, excitation_wavelength,N_c, excitation_P , alpha_nr, d_E,alpha_isc, activation_P=None ):
    
    if activation_P is None:
        k_01 = excitation_rate(epsilon, excitation_P, excitation_wavelength) 
        k_10 = 1/excitation_lifetime 
        k_1T = alpha_isc * np.exp(-d_E)
        k_T0 = alpha_nr * k_10
        k_1B = 1/(N_c * excitation_lifetime) 
        
        return k_01, k_10, k_1T, k_T0, k_1B 
        
    else :
        k_na = activation_rate(activation_P)
        k_01 = excitation_rate(epsilon, excitation_P, excitation_wavelength) 
        k_10 = 1/excitation_lifetime 
        k_1T = alpha_isc * np.exp(-d_E)
        k_T0 = alpha_nr * k_10
        k_1B = 1/(N_c * excitation_lifetime) 
        
        return k_na, k_01, k_10, k_1T, k_T0, k_1B


def activation_rate(activation):

    """activation: a dict with all the informations about activation process
    """
    return activation["P"] * activation["beta"]

    
    