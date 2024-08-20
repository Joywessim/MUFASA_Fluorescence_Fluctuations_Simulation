import numpy as np





def apply_photoswitching_one_molecule(time_points, trans_p, initial_state=0, scheme=None):
    fluorophore_states = np.zeros(int(time_points), dtype=np.int8)
    cumulative_trans_p = np.cumsum(trans_p, axis=1)
    
    states = np.zeros(int(time_points), dtype=np.int8)
    states[0] = initial_state if scheme == "blinking" else initial_state + 1

    t = 1
    while t < int(time_points) and states[t - 1] != 4:
        current_state = states[t - 1]
        threshold = np.random.rand()
        next_state = np.searchsorted(cumulative_trans_p[current_state], threshold, side="right")

        states[t] = next_state if next_state < trans_p.shape[1] else current_state
        t += 1

    states[t:] = 4  # fill with bleached
    fluorophore_states = states - 1  # to get -1, 0, 1, 2, 3

    return fluorophore_states, t


def count_cycles_per_frame(states, time_points, num_frames):
    frames = states.reshape((num_frames, int(time_points//num_frames)))
    
    # Identify where states transition from 0 to 1
    transitions_to_1 = (frames[:, :-1] == 0) & (frames[:, 1:] == 1)
    
    # Identify where states transition from 1 to 0
    transitions_to_0 = (frames[:, :-1] == 1) & (frames[:, 1:] == 0)
    
    # Shift the transitions_to_1 by one to align with transitions_to_0
    transitions_to_1_shifted = np.pad(transitions_to_1[:, :-1], ((0, 0), (1, 0)), mode='constant')
    
    # Count complete cycles: where both parts of the transitions are true
    complete_cycles = transitions_to_1_shifted & transitions_to_0
    photon_cycles_per_frame = np.sum(complete_cycles, axis=1)
    
    return photon_cycles_per_frame


































                                                                                
                                                                                
                                                                                
                                                                                
            #             ,@@@/../@@@*              @@@(,.*@@@(                   
            #           @@            @@         @@            @@                 
            #          @,              ,@@@@(&@@@@               @*               
            # #@,#@  ,@@                @&     /@                %@,.             
            # @       #@                @%     .@                &@               
            # @.       &@              @@       (@.             @@                
            #            @@/        *@@           &@@        .@@                  
            #   .  &         #@@@@%                   (@@@@@.                     
                                                                                
            #            @             .@@(     @@(             .@                
            #            @@*        @@@@@@@@@@@@@@@@@@@        &@&                
            #            %@@@*  ,@@@@@@@@@@@@@@@@@@@@@@@@@.  (@@@.                
            #             @@@@@@@@@@@@@@@#       @@@@@@@@@@@@@@@@                 
            #             *@@@@@@@         @@@@@        ,@@@@@@@                  
            #              @@@@@@@@       @@@@@@@       @@@@@@@,                  
            #               #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                    
            #                 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                     
            #                  /@@@@@@@@@@@@@@@@@@@@@@@@@@@                       
            #                    .@@@@@@@@@@@@@@@@@@@@@@@                         
            #                        @@@@@@@@@@@@@@@@&                            
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
























