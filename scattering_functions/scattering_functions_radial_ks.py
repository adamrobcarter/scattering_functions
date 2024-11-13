import numpy as np
import multiprocessing
import functools
import tqdm
import warnings

def get_particles_at_frame(particles):
    num_timesteps = int(particles[:, 2].max()) + 1

    # data is x,y,t
    particles[:, 2] -= particles[:, 2].min() # convert time to being 0-based

    # first find max number of particles at any one timestep
    num_particles_at_frame = np.bincount(particles[:, 2].astype('int'))
    max_particles_at_frame = num_particles_at_frame.max()
    print('max particles at any one frame', max_particles_at_frame)

    # for F (not self), we don't need the ID, so we just provide a list of particles
    # some datasets may already have the ID in column 4, so we only select the first 3 columns
    # however this operation is memory-expensive for very large datasets
    if particles.size > 18e8:
        assert particles.shape[1] == 3
    else:
        particles = particles[:, [0, 1, 2]]

    # this does what the below does but just slower
    # particles_at_frame = np.full((num_timesteps, max_particles_at_frame, 2), np.nan)
    # num_done_per_timestep = np.zeros(num_timesteps, dtype='int')
    # for row in tqdm.tqdm(particles):
    #     timestep = int(row[2])
    #     particles_at_frame[timestep, num_done_per_timestep[timestep], :] = (row[0], row[1])
    #     num_done_per_timestep[timestep] += 1

    # the below is a heavily-optimised method for turning the array of particles
    # into an array that is (num timesteps) x (max particles per timestep) x 2
    # we add extra nan rows such that each timestep has the same number of rows
    num_extra_rows = (max_particles_at_frame - num_particles_at_frame).sum()
    extra_rows = np.full((num_extra_rows, 3), np.nan)
    num_rows_added = 0
    for frame in tqdm.trange(num_timesteps, desc='reshaping', leave=False):
        for i in range(max_particles_at_frame-num_particles_at_frame[frame]):
            extra_rows[num_rows_added, 2] = frame
            num_rows_added += 1

    all_rows = np.concatenate((particles, extra_rows), dtype=particles.dtype, axis=0)
    del particles, extra_rows

    # then by sorting and reshaping, we can get the structure we want
    all_rows = all_rows[all_rows[:, 2].argsort()]
    # all_rows.view('f4,f4,f4').sort(order=['f2'], axis=0) # this is a way of sorting the array by the 3rd (f2) column in place, saves loads of ram but takes much longer
    particles_at_frame = all_rows.reshape((num_timesteps, max_particles_at_frame, 3))
    del all_rows
    # now remove the time column, leaving just x and y
    particles_at_frame = particles_at_frame[:, :, [0, 1]]

    return particles_at_frame, num_timesteps

def intermediate_scattering(k, num_k_angles, max_time_origins, d_frames, particles_at_frame, num_timesteps, cores):
    """
    log: if True, the k bins will be logarithmically spaced
    F_type: 'F' or 'F_s' for the self-ISF. The self-ISF code is probably broken atm
    num_k_bins: number of k points. Computation time is proportional to this squared
    max_time_origins: averages will be taken over this many different time origins. Computation time is directly proportional to this
    d_frames: an array of values of delta (frame number) that will be calculated (s.t. delta t is this multiplied by the timestep)
    particles: list of rows of (x, y, t) or (x, y, t, #)
    max_K:
    min_K:
    """
    assert 0 in d_frames, 'you need 0 in d_frames in order to calculate S(k) for the normalisation'

    k = np.array(k)
    num_k_mags = k.size

    k_x, k_y = get_k_for_intermediate_scattering(k, num_k_angles)

    Fs_full     = np.full((len(d_frames), num_k_mags, num_k_angles), np.nan)
    F_uncs_full = np.full((len(d_frames), num_k_mags, num_k_angles), np.nan)
    # ks_full     = np.full((len(d_frames), num_k_mags, num_k_angles), np.nan)

    # first find the particles at each timestep, otherwise we're transferring
    # the whole of data to each process
    print('finding particles at each timestep')


    assert np.all(d_frames < num_timesteps)

    use_every_nth_frame = max(int(num_timesteps / max_time_origins), 1)

    if use_every_nth_frame > 1:
        warnings.warn(f'Using every {use_every_nth_frame}th frame as a time origin. Eventually you may want to use every frame')

    print('beginning computation')
    print('particles_at_frame:', common.arraysize(particles_at_frame))

    progress = tqdm.tqdm(total=len(d_frames)*num_timesteps//use_every_nth_frame)

    if cores > 16:
        warnings.warn(f'using {cores} cores')

    with multiprocessing.Pool(cores) as pool:
        # for dframe_i in tqdm.trange(len(d_frames)):
        for dframe_i in range(len(d_frames)):
            F_unbinned, F_unc_unbinned = intermediate_scattering_for_dframe(dframe_i,
                                use_every_nth_frame=use_every_nth_frame, d_frames=d_frames, particles_at_frame=particles_at_frame,
                                num_frames=num_timesteps, k_x=k_x, k_y=k_y, pool=pool, progress=progress)
            
            Fs_full    [dframe_i, :, :] = F_unbinned
            F_uncs_full[dframe_i, :, :] = F_unc_unbinned

            # assert np.all(F_unbinned == F_unbinned[0])
            # print(F_unbinned.mean(axis=1))

    Fs    = Fs_full    .mean(axis=2) # average over theta
    F_unc = F_uncs_full.mean(axis=2) # average over theta

    print(Fs)

    return Fs, F_unc, Fs_full, F_uncs_full, k_x, k_y

def intermediate_scattering_for_dframe(dframe_i, use_every_nth_frame, d_frames, particles_at_frame, num_frames, k_x, k_y, pool, progress):
    d_frame = int(d_frames[dframe_i])

    assert num_frames > d_frame, f'd_frame={d_frame}, num_frames={num_frames}'

    frames_to_use = range(0, num_frames-d_frame-1, use_every_nth_frame)

    assert max(frames_to_use) + d_frame < num_frames
    
    num_used_frames = len(frames_to_use)

    # F = np.full((num_used_frames, k_bins.size-1), np.nan)
    # k = np.full((k_bins.size-1),                  np.nan) # +1 b/c we get the left and right of the final bin
    F_full     = np.full((num_used_frames, *k_x.shape), np.nan)
    F_unc_full = np.full((num_used_frames, *k_x.shape), np.nan)
    # k_full     = np.full((k_x.size, k_y.size),                  np.nan)

    parallel = True
    if parallel:
        bound = functools.partial(intermediate_scattering_preprocess_run_postprocess,
                                    k_x, k_y, intermediate_scattering_internal)
        
        particles = []
        # for frame_index in tqdm.trange(num_used_frames, desc='preparing data', leave=False):
        for frame_index in range(num_used_frames):
            frame = int(frames_to_use[frame_index])
            particles.append((particles_at_frame[frame, :, :], particles_at_frame[frame+d_frame, :, :]))

        # print('passing', common.arraysize(particles[0][0], 2), 'to each process')

        # results = pool.map(bound, particles, chunksize=1)
        # progress.update(len(results))

        results = []
        tasks = pool.imap(bound, particles, chunksize=1)
        for result in tasks:
            results.append(result)
            progress.update()

        # results is now (num used frames) x 2 x (len of slice) <- OUT OF DATE?
        for i, result in enumerate(results):
            F_unbinned = result
            F_full[i, :, :] = F_unbinned

    # need nanmean because binned_statistic will return nan if the bin is empty
    return np.nanmean(F_full, axis=0), np.nanstd(F_full, axis=0)/np.sqrt(num_used_frames)


def intermediate_scattering_preprocess_run_postprocess(k_x, k_y, func, particles):
    particles_t0, particles_t1 = preprocess_scattering(particles[0], particles[1])
    F_unbinned = func(particles_t0, particles_t1, k_x, k_y)
    
    assert np.isnan(F_unbinned).sum() == 0, f'F was {np.isnan(F_unbinned).sum()/F_unbinned.size*100:.0f}% NaN'
    
    return F_unbinned

def preprocess_scattering(particles_t0, particles_t1):
    # first remove any nan particles
    t0_nans = np.any(np.isnan(particles_t0), axis=1)
    t1_nans = np.any(np.isnan(particles_t1), axis=1)
    # # nans = t0_nans | t1_nans
    # # print(f'missing particles: {nans.sum()/nans.size*100}%')
    particles_t0 = particles_t0[~t0_nans, :]
    particles_t1 = particles_t1[~t1_nans, :]

    # assert np.isnan(particles_t0).sum() == 0
    # assert np.isnan(particles_t1).sum() == 0

    return particles_t0, particles_t1

def get_k_for_intermediate_scattering(k, num_angles):

    theta = np.linspace(0, np.pi, num=num_angles, endpoint=False)
    theta = np.linspace(-np.pi, np.pi, num=num_angles, endpoint=False)
    theta = theta[np.newaxis, :]

    k_ = k[:, np.newaxis]

    k_x = k_ * np.sin(theta)
    k_y = k_ * np.cos(theta)

    return k_x, k_y

def intermediate_scattering_internal(particles_t0, particles_t1, k_x, k_y):
    # Thorneywork et al 2018 eq (27))
    
    particle_t0_x = particles_t0[:, 0]
    particle_t0_y = particles_t0[:, 1]
    particle_t1_x = particles_t1[:, 0]
    particle_t1_y = particles_t1[:, 1]

    # dimensions are
    # mu x nu x kx x ky
    x_mu = particle_t0_x[:, np.newaxis, np.newaxis, np.newaxis]
    y_mu = particle_t0_y[:, np.newaxis, np.newaxis, np.newaxis]
    x_nu = particle_t1_x[np.newaxis, :, np.newaxis, np.newaxis]
    y_nu = particle_t1_y[np.newaxis, :, np.newaxis, np.newaxis]

    k_x = k_x[np.newaxis, np.newaxis, :, :]
    k_y = k_y[np.newaxis, np.newaxis, :, :]

    k_dot_r_mu = np.multiply(k_x, x_mu, dtype='float64') + np.multiply(k_y, y_mu, dtype='float64')
    k_dot_r_nu = np.multiply(k_x, x_nu, dtype='float64') + np.multiply(k_y, y_nu, dtype='float64')
    # print(k_x.mean(), k_y.mean(), x_mu.mean(), x_nu.mean(), y_mu.mean(), y_nu.mean(), k_dot_r_mu.mean(), k_dot_r_nu.mean())
    # k_dot_r_mu = k_x * x_mu + k_y * y_mu
    # k_dot_r_nu = k_x * x_nu + k_y * y_nu

    # print(np.cos(k_dot_r_mu).sum(axis=(0, 1)).shape, np.cos(k_dot_r_mu).sum(axis=(0)).shape)
    cos_term1 = np.cos(k_dot_r_mu).sum(axis=0) # sum over mu
    cos_term2 = np.cos(k_dot_r_nu).sum(axis=1) # sum over nu
    cos_accum = cos_term1 * cos_term2
    del cos_term1, cos_term2
    
    sin_term1 = np.sin(k_dot_r_mu).sum(axis=0)
    sin_term2 = np.sin(k_dot_r_nu).sum(axis=1)
    sin_accum = sin_term1 * sin_term2
    del sin_term1, sin_term2
    del k_dot_r_mu, k_dot_r_nu
    
    num_particles = (particles_t0.shape[0] + particles_t1.shape[0]) / 2
    if num_particles == 0:
        warnings.warn('found no particles in either timestep')
        contrib = np.zeros_like(cos_accum)
    else:
        contrib = 1/num_particles * ( cos_accum + sin_accum )
    # del cos_accum, sin_accum # probably unneeded

    return contrib

# def distinct_intermediate_scattering_internal(particles_t0, particles_t1, k_x, k_y):
    
#     particle_t0_x = particles_t0[:, 0]
#     particle_t0_y = particles_t0[:, 1]
#     particle_t1_x = particles_t1[:, 0]
#     particle_t1_y = particles_t1[:, 1]

#     # dimensions are
#     # mu x nu x kx x ky
#     x_mu = particle_t0_x[:, np.newaxis, np.newaxis, np.newaxis]
#     y_mu = particle_t0_y[:, np.newaxis, np.newaxis, np.newaxis]
#     x_nu = particle_t1_x[np.newaxis, :, np.newaxis, np.newaxis]
#     y_nu = particle_t1_y[np.newaxis, :, np.newaxis, np.newaxis]

#     k_x = k_x[np.newaxis, np.newaxis, :, np.newaxis]
#     k_y = k_y[np.newaxis, np.newaxis, np.newaxis, :]
    
#     # TODO: do we not need to consider negative n?!!
#     # actually I think not because we already consider u -> v and v -> u

#     k_dot_r_mu = np.multiply(k_x, x_mu, dtype='float64') + np.multiply(k_y, y_mu, dtype='float64')
#     k_dot_r_nu = np.multiply(k_x, x_nu, dtype='float64') + np.multiply(k_y, y_nu, dtype='float64')
#     # print(k_x.mean(), k_y.mean(), x_mu.mean(), x_nu.mean(), y_mu.mean(), y_nu.mean(), k_dot_r_mu.mean(), k_dot_r_nu.mean())
#     # k_dot_r_mu = k_x * x_mu + k_y * y_mu
#     # k_dot_r_nu = k_x * x_nu + k_y * y_nu

#     # print(np.cos(k_dot_r_mu).sum(axis=(0, 1)).shape, np.cos(k_dot_r_mu).sum(axis=(0)).shape)
#     cos_term1 = np.cos(k_dot_r_mu).sum(axis=0) # sum over mu
#     cos_term2 = np.cos(k_dot_r_nu).sum(axis=1) # sum over nu
#     cos_accum = cos_term1 * cos_term2
#     del cos_term1, cos_term2
    
#     sin_term1 = np.sin(k_dot_r_mu).sum(axis=0)
#     sin_term2 = np.sin(k_dot_r_nu).sum(axis=1)
#     sin_accum = sin_term1 * sin_term2
#     del sin_term1, sin_term2
#     del k_dot_r_mu, k_dot_r_nu
    
#     num_particles = (particles_t0.shape[0] + particles_t1.shape[0]) / 2
#     if num_particles == 0:
#         warnings.warn('found no particles in either timestep')
#         contrib = np.zeros_like(cos_accum)
#     else:
#         contrib = 1/num_particles * ( cos_accum + sin_accum )
#     k = np.sqrt(k_x**2 + k_y**2)
#     # del cos_accum, sin_accum # probably unneeded

#     return k, contrib