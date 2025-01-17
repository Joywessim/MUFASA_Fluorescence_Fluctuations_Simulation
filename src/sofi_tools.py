import numpy  as np
from scipy.interpolate import interp1d


def generate_trace(Ion, Ton, Toff, Tbl, frames, dynamics = False,seed=None):
    
    if seed is not None:
        np.random.seed(seed)

    times, Ton,  bleaching_time, Tbl = generate_times(Ton, Toff,Tbl,frames)

    photons = np.vstack((np.zeros_like(Ton), Ton * Ion))
    photons = photons.flatten('F')
    photons = np.cumsum(photons)
    interpolator = interp1d(times, photons, kind='linear', fill_value=0, bounds_error=False)
    new_time_points = np.arange(0, frames+1 )
    interpolated_photons = interpolator(new_time_points)
    photons=  np.diff(interpolated_photons)


    if dynamics:
        # Continuous time vector
        total_time = times[-1]  # Total duration
        time_vector = np.linspace(0, total_time, 10000)  # Fine time vector for continuous time representation

        # Initialize the state vector for continuous time
        state_vector = np.zeros_like(time_vector, dtype=int)
        # Loop through the times array to set the state
        for i in range(0, len(times), 2):
            start_off = times[i]
            end_off = times[i + 1] if i + 1 < len(times) else total_time

            state_vector[(time_vector >= start_off) & (time_vector < end_off)] = 1  # 'On' period

            if i + 1 < len(times):
                start_on = times[i + 1]
                end_on = times[i + 2] if i + 2 < len(times) else total_time

                state_vector[(time_vector >= start_on) & (time_vector < end_on)] = 0  # 'Off' period

        # Handle bleaching in the state_vector
        if bleaching_time > 0:
            bleached_time = times[2 * bleaching_time]
            state_vector[time_vector >= bleached_time] = -1  # Set to -1 after bleaching

        return times,bleaching_time,photons,state_vector
    else:
        return times,bleaching_time,photons


def generate_times(Ton, Toff,Tbl, frames):
    cycle = Ton + Toff  # length of a cycle: for a fluorophore to reach the on-state and in the off-state
    cycles = 10 + int(np.ceil(frames / cycle))  # number of cycles in the entire experiment (the +10 creates ten cycles in addition to avoid problems near t~0)
    times = np.vstack([-Toff * np.log(np.random.rand(cycles)), -Ton * np.log(np.random.rand(cycles))])
    times[0,0] = times[0,0] - np.random.rand() * np.sum(times.flatten('F')[:10])
    times = times.flatten('F')  # equivalent to MATLAB's times(:)
    times = np.cumsum(times)
    # while times[-1] < frames:
    #     cycles = int(np.ceil(2*(frames - times[-1]) / cycle))
    #     cycles = np.vstack([-Toff * np.log(np.random.rand(cycles)), -Ton * np.log(np.random.rand(cycles))])
    #     cycles[0,0] += times[-1]
    #     times = np.vstack(times,np.cumsum((cycles.flatten('F'))))   

    times = times.T      
    Ton = times[1::2] - times[:-1:2]
    Tbl = np.cumsum(Ton) + Tbl * np.log(np.random.rand())
    n= np.where(Tbl>0)[0]
    if n.shape[0]>0:
        Ton[n[1:]] = 0
        bleaching_time = n[0]
        Ton[bleaching_time] = Ton[bleaching_time] - Tbl[bleaching_time]
        times[2*bleaching_time] = times[2*bleaching_time] - Tbl[bleaching_time]
    else:
        bleaching_time = 0
    return times, Ton,  bleaching_time, Tbl


