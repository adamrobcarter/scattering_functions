import numpy as np
import scipy.stats
import multiprocessing
import functools
import tqdm
import warnings, time

def get_particles_at_frame(F_type, particles):
    num_timesteps = int(particles[:, 2].max()) + 1

    # data is x,y,t
    particles[:, 2] -= particles[:, 2].min() # convert time to being 0-based

    # first find max number of particles at any one timestep
    num_particles_at_frame = np.bincount(particles[:, 2].astype('int'), minlength=num_timesteps)
    max_particles_at_frame = num_particles_at_frame.max()
    print('max particles at any one frame', max_particles_at_frame)
    assert num_particles_at_frame.size == num_timesteps

    if F_type == 'F_s':
        # for Fself, we need the IDs, so we provide a list where the nth element is the nth particle
        assert particles.shape[1] == 4, 'for self intermediate scattering, you should provide rows of x,y,t,#'

        # we sort the ting by particle ID, then time
        print('sorting')
        particles = particles[particles[:, 3].argsort()]
        particles = particles[np.lexsort((particles[:, 2], particles[:, 3]))] # sort by ID then time
        print('sorted')

        # find trajectories that are long enough
        i = 0
        MIN_TRAJ_LENGTH = 50
        good_ids = []
        progress = tqdm.tqdm(total=particles.shape[0], desc='finding trajectories')
        skipped = 0
        while i < particles.shape[0]-1:
            start_i = i
            current_id = int(particles[i, 3])
            start_time = particles[i, 2]
            while i < particles.shape[0]-1 and particles[i, 3] == current_id:
                i += 1
                progress.update()
            end_i = i
            # print(particles[start_i:end_i, :])
            end_time = particles[i-1, 2]
            # print(f'end_t={end_time}, start_t={start_time} i={i}')
            assert end_time >= start_time, f'end_t={end_time}, start_t={start_time}'
            traj_length = end_time - start_time
            rows_used = end_i - start_i
            assert end_i > start_i
            # print(rows_used, traj_length, rows_used == traj_length + 1)
            if rows_used != traj_length + 1: # this essentially checks that t is going up by one each time
                skipped += 1
                # print()
            else:
                if traj_length > MIN_TRAJ_LENGTH:
                    good_ids.append((current_id, start_i, end_i))
            
            i += 1

            # if i > 10:
            #     break

        progress.close()

        # import sys
        # sys.exit()
        print('good trajs', len(good_ids)/particles[:, 3].max())
        print('skipped', skipped/particles[:, 3].max())

        # save these trajectories in a new shape
        particles_at_frame = np.full((num_timesteps, len(good_ids), 2), np.nan)
        xys = particles[:, [0, 1]]
        for j in tqdm.trange(len(good_ids), desc='reshaping'):
            old_id = good_ids[j][0]
            new_id = j # need to resample the IDs when we remove some particles
            start_t = int(particles[good_ids[j][1],   2])
            end_t   = int(particles[good_ids[j][2]-1, 2])
            # print('ids', good_ids[j][1], good_ids[j][2], start_t, end_t, id)
            particles[good_ids[j][1]:good_ids[j][2], :]
            # print('p_a_f shape', particles_at_frame[start_t:end_t+1, id, :].shape)
            xys[good_ids[j][1]:good_ids[j][2], :]
            particles_at_frame[start_t:end_t+1, new_id, :]
            particles_at_frame[start_t:end_t+1, new_id, :] = xys[good_ids[j][1]:good_ids[j][2], :]

            # if i > 10:
            #     break

        num_particles = int(particles[:, 3].max()) + 1
        assert num_particles > 0

        del particles

    else:
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

def intermediate_scattering(F_type, num_k_bins, max_time_origins, d_frames, particles_at_frame, num_timesteps, max_K, min_K, cores, use_zero=True, use_big_k=True, linear_log_crossover_k=1, use_doublesided_k=False):
    """
    F_type: 'F' or 'F_s' for the self-ISF. The self-ISF code is probably broken atm
    num_k_bins: number of k points. Computation time is proportional to this squared
    max_time_origins: averages will be taken over this many different time origins. Computation time is directly proportional to this
    d_frames: an array of values of delta (frame number) that will be calculated (s.t. delta t is this multiplied by the timestep)
    particles_at_frame: list of lists of (x, y) for each timesep
    max_K:
    min_K: 2d list/tuple/ndarray as (min_K_x, min_K_y)
    """
    assert not np.isnan(max_K)
    assert 0 in d_frames, 'you need 0 in d_frames in order to calculate S(k) for the normalisation'
    d_frames = np.array(d_frames) # (possible) list to ndarray

    k_x, k_y, k_bins = get_k_and_bins_for_intermediate_scattering(min_K, max_K, num_k_bins, use_zero=use_zero, use_big_k=use_big_k, linear_log_crossover_k=linear_log_crossover_k, use_doublesided_k=use_doublesided_k)
    
    num_k_bins = k_bins.size - 1

    Fs    = np.full((len(d_frames), num_k_bins), np.nan)
    F_unc = np.full((len(d_frames), num_k_bins), np.nan)
    # print(f'F size {common.arraysize(Fs)}')
    ks   = np.full((len(d_frames), num_k_bins), np.nan)

    Fs_full     = np.full((len(d_frames), k_x.size, k_y.size), np.nan)
    F_uncs_full = np.full((len(d_frames), k_x.size, k_y.size), np.nan)
    ks_full     = np.full((len(d_frames), k_x.size, k_y.size), np.nan)

    # first find the particles at each timestep, otherwise we're transferring
    # the whole of data to each process
    print('finding particles at each timestep')


    assert np.all(d_frames <= num_timesteps)

    use_every_nth_frame = max(int(num_timesteps / max_time_origins), 1)
    
    if use_every_nth_frame > 1:
        warnings.warn(f'Using every {use_every_nth_frame}th frame as a time origin. Eventually you may want to use every frame')
    else:
        print('using every frame as a time origin')

    print('beginning computation')

    progress = tqdm.tqdm(total=len(d_frames)*num_timesteps//use_every_nth_frame, smoothing=0.03) # low smoothing makes it more like average speed and less like instantaneous speed

    if cores > 16:
        warnings.warn(f'using {cores} cores')

    if cores > 1:
        pool = multiprocessing.Pool(cores)
    else:
        pool = None
        
    for dframe_i in range(len(d_frames)):
        F_, F_unc_, k_, F_unbinned, F_unc_unbinned, k_unbinned = intermediate_scattering_for_dframe(dframe_i, F_type=F_type,
                            use_every_nth_frame=use_every_nth_frame, d_frames=d_frames, particles_at_frame=particles_at_frame,
                            num_frames=num_timesteps, k_x=k_x, k_y=k_y, k_bins=k_bins, pool=pool, progress=progress)
        
        Fs   [dframe_i, :] = F_
        F_unc[dframe_i, :] = F_unc_
        ks   [dframe_i, :] = k_
        Fs_full    [dframe_i, :, :] = F_unbinned
        F_uncs_full[dframe_i, :, :] = F_unc_unbinned
        ks_full    [dframe_i, :, :] = k_unbinned

    if cores > 1:
        pool.close()


    return Fs, F_unc, ks, Fs_full, F_uncs_full, ks_full, k_x, k_y

def intermediate_scattering_for_dframe(dframe_i, F_type, use_every_nth_frame, d_frames, particles_at_frame, num_frames, k_x, k_y, k_bins, pool, progress):
    d_frame = int(d_frames[dframe_i])

    assert num_frames > d_frame, f'd_frame={d_frame}, num_frames={num_frames}'

    time_origins_to_use = range(0, num_frames-d_frame-1, use_every_nth_frame)
    assert len(time_origins_to_use) > 0

    assert max(time_origins_to_use) + d_frame < num_frames
    
    num_used_time_origins = len(time_origins_to_use)

    F = np.full((num_used_time_origins, k_bins.size-1), np.nan)
    k = np.full((k_bins.size-1),                  np.nan) # +1 b/c we get the left and right of the final bin
    F_full     = np.full((num_used_time_origins, k_x.size, k_y.size), np.nan)
    F_unc_full = np.full((num_used_time_origins, k_x.size, k_y.size), np.nan)
    k_full     = np.full((k_x.size, k_y.size),                  np.nan)
                    
    if F_type == 'F_s':
        func = self_intermediate_scattering_internal
    else:
        func = intermediate_scattering_internal

    
    bound = functools.partial(intermediate_scattering_preprocess_run_postprocess,
                                k_x, k_y, k_bins, func)
    
    particles = []
    # for frame_index in tqdm.trange(num_used_frames, desc='preparing data', leave=False):
    for frame_index in range(num_used_time_origins):
        time_origin = int(time_origins_to_use[frame_index])
        particles.append((particles_at_frame[time_origin, :, :], particles_at_frame[time_origin+d_frame, :, :]))

    # print('passing', common.arraysize(particles[0][0], 2), 'to each process')

    # results = pool.map(bound, particles, chunksize=1)
    # progress.update(len(results))

    if pool:
        results = []
        tasks = pool.imap(bound, particles, chunksize=1)
        for result in tasks:
            results.append(result)
            progress.update()
    else:
        results = map(bound, tqdm.tqdm(particles))

    # results is now (num used frames) x 2 x (len of slice)
    for i, result in enumerate(results):
        k_binned, F_binned, k_unbinned, F_unbinned = result
        F[i, :] = F_binned
        F_full[i, :, :] = F_unbinned

        if i == 0:
            k = k_binned
            k_full = k_unbinned
        else:
            assert np.array_equal(k, k_binned, equal_nan=True)
            assert np.array_equal(k_full, k_unbinned, equal_nan=True)

    # need nanmean because binned_statistic will return nan if the bin is empty
    return np.nanmean(F, axis=0), np.nanstd(F, axis=0)/np.sqrt(num_used_time_origins), k, np.nanmean(F_full, axis=0), np.nanstd(F_full, axis=0)/np.sqrt(num_used_time_origins), k_full


def intermediate_scattering_preprocess_run_postprocess(k_x, k_y, k_bins, func, particles):
    particles_t0, particles_t1 = preprocess_scattering(particles[0], particles[1])
    k_unbinned, F_unbinned = func(particles_t0, particles_t1, k_x, k_y)
    
    assert np.isnan(F_unbinned).sum() <= 1, f'F was {np.isnan(F_unbinned).sum()/F_unbinned.size*100:.0f}% NaN'
    # one nan point is allowed at k=(0, 0)
    
    k_binned, F_binned = postprocess_scattering(k_unbinned, F_unbinned, k_bins)
    return k_binned, F_binned, k_unbinned, F_unbinned

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

def postprocess_scattering(k, F, k_bins):
    # any nan points in f we remove, and we remove them from k too. These nans are the ones we set during the calculation
    F_flat = F.flatten()
    k_flat = k.flatten()
    nans = np.isnan(F_flat)

    F_binned, _, _ = scipy.stats.binned_statistic(k_flat[~nans], F_flat[~nans], 'mean', bins=k_bins)
    k_binned, _, _ = scipy.stats.binned_statistic(k_flat[~nans], k_flat[~nans], 'mean', bins=k_bins)
    # ^^ we used to use the middle of the bin, but this will be slightly skewed for small k, so we do a proper average
    # binned statistic returns NaN if the bin is empty
    
    assert np.isnan(F_binned).sum() < F_binned.size

    return k_binned, F_binned

def get_k_and_bins_for_intermediate_scattering(min_K, max_K, num_k_bins, use_zero, use_big_k=True, linear_log_crossover_k=1, use_doublesided_k=False):
    assert use_zero

    if True:
        k_small_x = np.arange(min_K[0], linear_log_crossover_k, min_K[0], dtype=np.float32) # these used to be f64 but idk why
        k_small_y = np.arange(min_K[1], linear_log_crossover_k, min_K[1], dtype=np.float32) 
        k_big = np.logspace(np.log10(linear_log_crossover_k), np.log10(max_K), num_k_bins, dtype=np.float32)
        
        if use_big_k:
            k_oneside_x = np.concatenate([k_small_x, k_big])
            k_oneside_y = np.concatenate([k_small_y, k_big])
        else:
            k_oneside_x = k_small_x
            k_oneside_y = k_small_y

        if use_zero:
            k_x = np.concatenate([-k_oneside_x[::-1], (0,), k_oneside_x])
        else:
            k_x = np.concatenate([-k_oneside_x[::-1], k_oneside_x])

        if use_doublesided_k:
            if use_zero:
                k_y = np.concatenate([-k_oneside_y[::-1], (0,), k_oneside_y])
            else:
                k_y = np.concatenate([-k_oneside_y[::-1], k_oneside_y])

        else:
            k_y = np.concatenate([(0,), k_oneside_y])

        # bin_edges = np.concatenate([(0,), np.logspace(np.log10(min_K), np.log10(max_K), num_k_bins)])
        bin_edges = np.concatenate([(0,), k_oneside_x])

    assert np.isnan(k_x).sum() == 0
    assert np.isnan(k_y).sum() == 0
    
    # assert np.isclose(k_x.mean(), 0), f'k_x.mean() = {k_x.mean()}'

    # there's a small problem here as we double count the points on k_x = 0 when on not doublesided mode

    return k_x, k_y, bin_edges

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
    # print('x dtype', x_mu.dtype)

    k_x_ = k_x[:, np.newaxis]
    k_y_ = k_y[np.newaxis, :]
    k = np.sqrt(k_x_**2 + k_y_**2)

    # we gonna set the k=(0, 0) point to nan later, for now we save where it is
    assert np.sum(k_x == 0) == 1, f'np.sum(k_x == 0) = {np.sum(k_x == 0)}'
    assert np.sum(k_y == 0) == 1, f'np.sum(k_y == 0) = {np.sum(k_y == 0)}'
    x_zero_index = np.argmax(k_x == 0)
    y_zero_index = np.argmax(k_y == 0)
    assert k_x[x_zero_index] == 0
    assert k_y[y_zero_index] == 0

    k_x = k_x[np.newaxis, np.newaxis, :, np.newaxis]
    k_y = k_y[np.newaxis, np.newaxis, np.newaxis, :]
    
    # TODO: do we not need to consider negative n?!!
    # actually I think not because we already consider u -> v and v -> u

    k_dot_r_mu = np.multiply(k_x, x_mu, dtype=np.float32) + np.multiply(k_y, y_mu, dtype=np.float32) # this used to by dtype=f64 but idk why
    k_dot_r_nu = np.multiply(k_x, x_nu, dtype=np.float32) + np.multiply(k_y, y_nu, dtype=np.float32)

    # set the k=(0, 0) point to nan
    k_dot_r_mu[:, :, x_zero_index, y_zero_index] = np.nan
    k_dot_r_nu[:, :, x_zero_index, y_zero_index] = np.nan



    # print(k_dot_r_mu.dtype)
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
        f = np.zeros_like(cos_accum)
    else:
        f = 1/num_particles * ( cos_accum + sin_accum )
    # del cos_accum, sin_accum # probably unneeded

    f = np.squeeze(f) # idk why by shape is currently (1, 132, 156)
    assert np.isnan(f[x_zero_index, y_zero_index])

    return k, f

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

def self_intermediate_scattering_internal(particles_t0, particles_t1, k_x, k_y):
    # remove particles that were nan in both sets
    nans = np.isnan(particles_t0[:, 0]) | np.isnan(particles_t1[:, 0])

    print(nans.shape, particles_t0.shape, particles_t1.shape)
    
    particles_t0 = particles_t0[~nans, :]
    particles_t1 = particles_t1[~nans, :]
    
    num_particles = particles_t0.shape[0]
    #print(f"kept {num_particles} of {num_particles_before}")
    
    particle_t0_x = particles_t0[:, 0]
    particle_t0_y = particles_t0[:, 1]
    particle_t1_x = particles_t1[:, 0]
    particle_t1_y = particles_t1[:, 1]

    # dimensions are
    # mu x kx x ky
    x_mu = particle_t0_x[:, np.newaxis, np.newaxis]
    y_mu = particle_t0_y[:, np.newaxis, np.newaxis]
    x_nu = particle_t1_x[:, np.newaxis, np.newaxis]
    y_nu = particle_t1_y[:, np.newaxis, np.newaxis]

    k_x = k_x[np.newaxis, :, np.newaxis]
    k_y = k_y[np.newaxis, np.newaxis, :]

    # k_dot_dr = k_x * (x_mu - x_nu)  +  k_y * (y_mu - y_nu)
    k_dot_dr = np.multiply(k_x, x_mu - x_nu, dtype='float32') + np.multiply(k_y, y_mu - y_nu, dtype='float32') # this used to be float64 idk why

    S = 1/num_particles * np.cos(k_dot_dr).sum(axis=(0))

    del k_dot_dr
    #neg = np.sum(contrib < 0)
    #print(f'{neg/contrib.size:.2f} negative')

    k = np.sqrt(k_x**2 + k_y**2)

    return k, S