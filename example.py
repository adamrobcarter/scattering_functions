import numpy as np
import scattering_functions

particles = np.array([
#    x    y    t
    [0.0, 0.0, 0],
    [1.0, 1.0, 0],
    [1.0, 1.0, 1],
    [2.0, 2.0, 1],
    [0.0, 0.0, 2],
    [1.0, 1.0, 2],
    [1.0, 1.0, 3],
    [2.0, 2.0, 3],
]).astype(np.float32)

particles_at_frame, times = scattering_functions.get_particles_at_frame('F', particles)

results = scattering_functions.intermediate_scattering(
    F_type = 'F',
    max_time_origins = 50,
    d_frames = [0, 1, 2],
    particles_at_frame = particles_at_frame,
    times = times,
    max_K = 10,
    min_K = (0.1, 0.1),
    cores = 16,
    num_k_bins = 50,
)
# results.F is F(k, t), shape (num d_frames) x (num k points)
# same for results.k, giving the k value for each point as above. Therefore, every row is identical

print('finished')