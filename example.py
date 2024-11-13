import numpy as np
import scattering_functions

particles = np.array([
#    x  y  t
    [0.0, 0.0, 0],
    [1.0, 1.0, 0],
    [1.0, 1.0, 1],
    [2.0, 2.0, 1],
    [0.0, 0.0, 2],
    [1.0, 1.0, 2],
    [1.0, 1.0, 3],
    [2.0, 2.0, 3],
])

particles_at_frame, num_timesteps = scattering_functions.get_particles_at_frame('F', particles)

Fs, F_unc, ks, F_unbinned, F_unc_unbinned, k_unbinned, k_x, k_y = scattering_functions.intermediate_scattering(
    F_type='F',
    max_time_origins=50, # max number of time origins to average over
    d_frames=[0, 1, 2], # times (in number of frames) to calculate f(k, t) at
    particles_at_frame=particles_at_frame,
    num_timesteps=num_timesteps,
    max_K=10,
    min_K=(0.1, 0.1), # (x, y)
    cores=16, # uses multiprocessing, set to 1 to turn of parallel excecution
    use_zero=True,
    num_k_bins=50, # num k points in the log k regime
    use_big_k=False, # skip the log k regime?
    linear_log_crossover_k=1, # k value at the crossover from the linear to log regime
    use_doublesided_k=True, # calculate the (redundant) other half of k-space
)
print('finished')
# on the k points:
# the ks are currently distributed linearly between 0 and linear_log_crossover with step size min_K
# and then logarithmically between linear_log_crossover and max_K with num_k_bins points