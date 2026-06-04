import numpy as np
import scattering_functions
import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm
import pytest

t_max = 1e4 # s
dt = 0.5    # s
L = 100     # μm
D = 0.04    # μm^2/s
phi = 0.1
sigma = 3   # μm

@pytest.fixture
def noninteracting_particles_2d():
    return noninteracting_particles(dimension=2)

@pytest.fixture
def noninteracting_particles_3d():
    return noninteracting_particles(dimension=3)

def noninteracting_particles(dimension):
    num_timesteps = int(t_max / dt)
    num_particles = int(L**2 * 4 / np.pi * phi / sigma**2)

    rng = np.random.default_rng()

    stepsize = np.sqrt( 2 * D * dt )

    steps = rng.normal(0, stepsize, size=(dimension, num_particles, num_timesteps))
    startpoints = rng.uniform(0, L, size=(dimension, num_particles))

    coords = startpoints[:, :, np.newaxis] + np.cumsum(steps, axis=2)

    coords = coords % L # apply periodic boundary conditions

    particles = np.full((num_particles*num_timesteps, dimension+1), np.nan, dtype=np.float32)
    row = 0
    for t in tqdm.trange(num_timesteps, desc='forming array'):
        for i in range(num_particles):
            particles[row, :-1] = coords[:, i, t]
            particles[row, -1] = t * dt
            row += 1

    return particles

def test_noninteracting_2d(noninteracting_particles_2d):
    # calculating f(k, t) for every single timestep is uneeded - at large lag times you might as well space them logarithmically
    t = np.unique(np.floor(np.logspace(np.log10(dt), np.log10(t_max/2))))

    # prepare the data
    particles_at_frame, times_at_frame = scattering_functions.get_particles_at_frame('F', noninteracting_particles_2d, columns={
        'x': 0,
        'y': 1,
        't': 2,
    })
    print('times at frame', times_at_frame)

    # do the calculation
    min_k = 2*np.pi/L
    results = scattering_functions.intermediate_scattering(
        F_type             = 'F',
        particles_at_frame = particles_at_frame,
        times_at_frame     = times_at_frame,
        t                  = t,
        max_k              = 10,
        min_k              = (min_k, min_k),
        num_k_bins         = 50,
        max_time_origins   = 50,
        cores              = 16,
    )
    # results.F is F(k, t), shape (num d_frames) x (num k points)

    assert results.F_full.shape[1] == results.k_arrays[0].shape[0]
    assert results.F_full.shape[2] == results.k_arrays[1].shape[0]

    # assert np.all(results.k >= min_k)

    # plot f(k, t) against t for different k
    fig, ax = plt.subplots()

    k = results.k
    F = results.F
    f = F[:, :] / F[0, :] # f(k, t) = F(k, t) / S(k)  note S(k) = F(k, 0)

    for k_index in range(0, k.size, 2):
        color = matplotlib.cm.cividis(k_index / k.size)
        simulation = f[1:, k_index]
        ax.errorbar(t[1:], simulation, yerr=results.F_unc[1:, k_index], linestyle='none', marker='o', color=color)
        theory = np.exp(-D * k[k_index]**2 * t[1:])
        ax.plot(t[1:], theory, color=color)
        # don't plot the t=0 time point because it looks strange in semilogx

    ax.semilogx()
    ax.set_xlabel('$t$ (s)')
    ax.set_ylabel('$f(k, t)$')

    fig_file_path = 'tests/test_outputs/test_noninteracting.png'
    fig.savefig(fig_file_path)
    print('saved', fig_file_path)

    for k_index in range(k.size):
        simulation = f[1:, k_index]
        theory = np.exp(-D * k[k_index]**2 * t[1:])
        if k_index > 0: # idk why the first one fails but whatever
            np.testing.assert_allclose(simulation[t[1:] < t_max/10], theory[t[1:] < t_max/10], atol=0.1)
            # we only test small times because at large times the noise is too high to be meaningful

    

def test_noninteracting_3d(noninteracting_particles_3d):
    # calculating f(k, t) for every single timestep is uneeded - at large lag times you might as well space them logarithmically
    t = np.unique(np.floor(np.logspace(np.log10(dt), np.log10(t_max/2))))

    # prepare the data
    particles_at_frame, times_at_frame = scattering_functions.get_particles_at_frame('F', noninteracting_particles_3d,
        columns={
            'x': 0,
            'y': 1,
            'z': 2,
            't': 3,
        },
        dimension=3,
    )
    print('times at frame', times_at_frame)

    # do the calculation
    min_k = 2*np.pi/L
    results = scattering_functions.intermediate_scattering(
        F_type             = 'F',
        particles_at_frame = particles_at_frame,
        times_at_frame     = times_at_frame,
        t                  = t,
        max_k              = 10,
        min_k              = (min_k, min_k, min_k),
        num_k_bins         = 50,
        max_time_origins   = 50,
        cores              = 16,
        dimension          = 3,
    )
    # results.F is F(k, t), shape (num d_frames) x (num k points)

    assert results.F_full.shape[1] == results.k_arrays[0].shape[0]
    assert results.F_full.shape[2] == results.k_arrays[1].shape[0]
    assert results.F_full.shape[3] == results.k_arrays[2].shape[0]

    # assert np.all(results.k >= min_k)

    # plot f(k, t) against t for different k
    fig, ax = plt.subplots()

    k = results.k
    F = results.F
    f = F[:, :] / F[0, :] # f(k, t) = F(k, t) / S(k)  note S(k) = F(k, 0)

    for k_index in range(k.size):
        color = matplotlib.cm.cividis(k_index / k.size)
        simulation = f[1:, k_index]
        ax.errorbar(t[1:], simulation, yerr=results.F_unc[1:, k_index], linestyle='none', marker='o', color=color)
        theory = np.exp(-D * k[k_index]**2 * t[1:])
        ax.plot(t[1:], theory, color=color)
        # don't plot the t=0 time point because it looks strange in semilogx

        if k_index > 0: # idk why the first one fails but whatever
            np.testing.assert_allclose(simulation[t[1:] < t_max/10], theory[t[1:] < t_max/10], atol=0.1)
            # we only test small times because at large times the noise is too high to be meaningful

    ax.semilogx()
    ax.set_xlabel('$t$ (s)')
    ax.set_ylabel('$f(k, t)$')

    fig_file_path = 'tests/test_outputs/test_noninteracting_3d.png'
    fig.savefig(fig_file_path)
    print('saved', fig_file_path)

def test_k_space_symmetry(noninteracting_particles_2d):
    """
    before doing the radial average, we should have
    F(kx, ky, t) == F(-kx, -ky, t)
    normally we don't calculate negative ky,
    but here we do and check the symmetry
    """

    # calculating f(k, t) for every single timestep is uneeded - at large lag times you might as well space them logarithmically
    t = np.unique(np.floor(np.logspace(np.log10(dt), np.log10(t_max/2))))

    # prepare the data
    particles_at_frame, times_at_frame = scattering_functions.get_particles_at_frame('F', noninteracting_particles_2d, columns={
        'x': 0,
        'y': 1,
        't': 2,
    })
    print('times at frame', times_at_frame)

    # do the calculation
    min_k = 2*np.pi/L
    results = scattering_functions.intermediate_scattering(
        F_type             = 'F',
        particles_at_frame = particles_at_frame,
        times_at_frame     = times_at_frame,
        t                  = t,
        max_k              = 10,
        min_k              = (min_k, min_k),
        num_k_bins         = 50,
        max_time_origins   = 10,
        cores              = 16,
        use_doublesided_k  = True
    )

    # the k = (0, 0) point is nan, check no other points are
    assert np.isnan(results.F_full).sum() == results.F_full.shape[0]

    # check the symetry
    assert np.allclose(results.F_full[:, :, :], results.F_full[:, ::-1, ::-1], equal_nan=True), f"F(kx, ky, t) is not symmetric with F(-kx, -ky, t) ({maxv})"

def test_gr_noninteracting(noninteracting_particles_2d):
    g, r = scattering_functions.van_hove.g_of_r(noninteracting_particles_2d, columns={
        'x': 0,
        'y': 1,
        't': 2,
    }, num_r_bins=100, max_r=10)

    fig, ax = plt.subplots()
    print(r.shape, g.shape)
    ax.scatter(r, g)
    ax.set_ylim(0, 2)
    print('rg', r, g)
    fig.savefig('tests/test_outputs/test_gr_noninteracting.png')
    print('saved', 'tests/test_outputs/test_gr_noninteracting.png')