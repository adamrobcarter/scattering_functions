import numpy as np
import scattering_functions
import tqdm

def generate_noninteracting_particles(L, phi, sigma, dt, D, max_t):
    num_timesteps = int(max_t / dt)
    num_particles = int(L**2 * 4 / np.pi * phi / sigma**2)

    rng = np.random.default_rng()

    stepsize = np.sqrt( 2 * D * dt )
    steps_x = rng.normal(0, stepsize, size=(num_particles, num_timesteps))
    startpoints_x = rng.uniform(0, L, size=(num_particles),              )
    steps_y = rng.normal(0, stepsize, size=(num_particles, num_timesteps))
    startpoints_y = rng.uniform(0, L, size=(num_particles),              )

    x = startpoints_x[:, np.newaxis] + np.cumsum(steps_x, axis=1)
    y = startpoints_y[:, np.newaxis] + np.cumsum(steps_y, axis=1)

    # trajs = np.full((num_particles*num_timesteps, 4), np.nan, dtype=np.float32)
    # row = 0
    # for t in tqdm.trange(num_timesteps, desc='forming array'):
    #     for i in range(num_particles):
    #         trajs[row, :] = [x[i, t], y[i, t], t, i]
    #         row += 1

    x = x % L
    y = y % L

    particles = np.full((num_particles*num_timesteps, 3), np.nan, dtype=np.float32)
    row = 0
    for t in tqdm.trange(num_timesteps, desc='forming array'):
        for i in range(num_particles):
            particles[row, :] = [x[i, t], y[i, t], t*dt]
            row += 1

    return particles

def test_noninteracting():
    # generate some test data
    t_max = 1e4 # s
    dt = 0.5    # s
    L = 100     # μm
    D = 0.04    # μm^2/s
    phi = 0.1
    sigma = 3   # μm
    particles = generate_noninteracting_particles(L, phi, sigma, dt, D, t_max)

    # calculating f(k, t) for every single timestep is uneeded - at large lag times you might as well space them logarithmically
    t = np.unique(np.floor(np.logspace(np.log10(dt), np.log10(t_max/2))))

    # prepare the data
    particles_at_frame, times_at_frame = scattering_functions.get_particles_at_frame('F', particles, columns={
        'x': 0,
        'y': 1,
        't': 2,
    })
    print('times at frame', times_at_frame)

    # do the calculation
    results = scattering_functions.intermediate_scattering(
        F_type             = 'F',
        particles_at_frame = particles_at_frame,
        times_at_frame     = times_at_frame,
        t                  = t.astype(type(times_at_frame[0])),
        max_k              = 10,
        min_k              = (2*np.pi/L, 2*np.pi/L),
        num_k_bins         = 50,
        max_time_origins   = 50,
        cores              = 16,
    )
    # results.F is F(k, t), shape (num d_frames) x (num k points)

    assert results.F_full.shape[1] == results.k_x.shape[0]
    assert results.F_full.shape[2] == results.k_y.shape[0]
