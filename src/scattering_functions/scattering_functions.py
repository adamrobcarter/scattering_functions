import numpy as np
import scipy.stats
import multiprocessing
import functools
import warnings
import collections

# we use tqdm for nice progress bars if it is available
try:
    import tqdm
    # progressbar = functools.partial(tqdm.tqdm, leave=False)
    progressbar = tqdm.tqdm
except ImportError:
    progressbar = lambda x, desc=None, total=None: x

def get_frames_with_delta_t(t, dt):
    # find all pairs of frames in `t` separated by `dt`
    pairs = []

    pairs = np.zeros((len(t), 2), dtype=int)
    num_pairs_found = 0

    # for frame1 in tqdm.trange(len(t), desc='finding frames with dt', leave=False):
    #     # print('a')
    #     for frame2 in range(frame1, len(t)):
    #         # print('b')
    #         if t[frame2] - t[frame1] == dt:
    #             # pairs.append([frame1, frame2])
    #             pairs[num_pairs_found, :] = [frame1, frame2]
    #             num_pairs_found += 1
    #             break

    for frame in range(len(t)):
        if len(index := np.where(t == t[frame] + dt)[0]):

            pairs[num_pairs_found, :] = (frame, index[0])
            num_pairs_found += 1

    pairs = pairs[:num_pairs_found, :]

    assert len(pairs), f'no frames with time delta {dt} were found in the data'

    assert np.isfinite(pairs).all()
    return pairs

def get_particles_at_frame(F_type, particles, columns):
    assert particles.dtype == np.float32
    assert isinstance(columns, dict), f'type(columns) == {type(columns)}'

    time_column = columns['t']

    # data is x,y,t
    particles[:, time_column] -= particles[:, time_column].min() # convert time to being 0-based

    # find max number of particles at any one timestep
    times, num_particles_at_frame = np.unique(particles[:, time_column], return_counts=True)
    num_timesteps = times.size
    max_particles_at_frame = num_particles_at_frame.max()
    assert num_particles_at_frame.size == num_timesteps, f'{num_particles_at_frame.size} != {num_timesteps}'
    assert max_particles_at_frame < 1.5 * num_particles_at_frame.mean(), f'max particles at frame {max_particles_at_frame} avg particles at frame {num_particles_at_frame.mean():.1f}'

    if F_type == 'F_s':
        assert False
        # for Fself, we need the IDs, so we provide a list where the nth element is the nth particle
        assert particles.shape[1] == 4, 'for self intermediate scattering, you should provide rows of x,y,t,#'

        # we sort the ting by particle ID, then time
        print('sorting')
        particles = particles[particles[:, 3].argsort()]
        particles = particles[np.lexsort((particles[:, time_column], particles[:, id_column]))] # sort by ID then time
        print('sorted')

        # find trajectories that are long enough
        i = 0
        MIN_TRAJ_LENGTH = 50
        good_ids = []
        progress = progressbar(total=particles.shape[0], desc='finding trajectories')
        skipped = 0
        while i < particles.shape[0]-1:
            start_i = i
            current_id = int(particles[i, id_column])
            start_time = particles[i, time_column]
            while i < particles.shape[0]-1 and particles[i, id_column] == current_id:
                i += 1
                progress.update()
            end_i = i
            # print(particles[start_i:end_i, :])
            end_time = particles[i-1, time_column]
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
        for j in progressbar(range(len(good_ids)), desc='reshaping for self'):
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

        num_particles = int(particles[:, id_column].max()) + 1
        assert num_particles > 0

        del particles

        # raise Exception('you have not yet implemented the new time (not frame) based approach here. if you implement it in MSD, be sure to back-copy')
        ############## actually it seems to just work...

    else:
        # for F (not self), we don't need the ID, so we just provide a list of particles
        # some datasets may already have the ID in column 4, so we only select the x,y,t columns
        particles = particles[:, [columns['x'], columns['y'], time_column]]
        # but in doing that, we changed the column indexes
        columns['x'] = 0
        columns['y'] = 1
        columns['t'] = 2
        time_column = 2

        # this does what the below does but just slower
        # note this may be out of date since the discontinuous times update
        # particles_at_frame = np.full((num_timesteps, max_particles_at_frame, 2), np.nan)
        # num_done_per_timestep = np.zeros(num_timesteps, dtype='int')
        # for row in tqdm.tqdm(particles):
        #     timestep = int(row[2])
        #     particles_at_frame[timestep, num_done_per_timestep[timestep], :] = (row[0], row[1])
        #     num_done_per_timestep[timestep] += 1

        # the below is an optimised method for turning the array of particles
        # into an array that is (num timesteps) x (max particles per timestep) x 2
        # we add extra nan rows such that each timestep has the same number of rows

        num_extra_rows = (max_particles_at_frame - num_particles_at_frame).sum()
        extra_rows = np.full((num_extra_rows, 3), np.nan, dtype=particles.dtype)
        assert extra_rows.size < 0.2 * particles.size, f'extra_rows.size = {extra_rows.size}, particles.size = {particles.size}'
        
        num_rows_added = 0
        for frame in progressbar(range(num_timesteps), desc='reshaping', disable=True):
            for i in range(max_particles_at_frame-num_particles_at_frame[frame]):
                extra_rows[num_rows_added, time_column] = times[frame]
                num_rows_added += 1

        all_rows = np.concatenate((particles, extra_rows), dtype=particles.dtype, axis=0)
        del particles, extra_rows
        # then by sorting and reshaping, we can get the structure we want
        all_rows = all_rows[all_rows[:, time_column].argsort()]
        # all_rows.view('f4,f4,f4').sort(order=['f2'], axis=0) # this is a way of sorting the array by the 3rd (f2) column in place, saves loads of ram but takes much longer
        particles_at_frame = all_rows.reshape((num_timesteps, max_particles_at_frame, 3))
        del all_rows
        # now remove the time column, leaving just x and y
        particles_at_frame = particles_at_frame[:, :, [columns['x'], columns['y']]]

    assert particles_at_frame.shape[0] == times.shape[0]
    assert np.isfinite(times).all()

    times = times.astype(np.int64) # was getting weird errors about this - note that issue Ryker flagged about the times_at_frame dtype

    return particles_at_frame, times

def intermediate_scattering(
        F_type, num_k_bins, max_time_origins, t, particles_at_frame, times_at_frame, max_k, min_k, cores=1,
        use_doublesided_k=False, Lx=None, Ly=None, window=None, quiet=False,
    ):
    assert np.isfinite(max_k)
    assert np.isfinite(times_at_frame).all()

    assert 0 in t, 'you need 0 in t in order to calculate S(k) for the normalisation'
    if max(t) > len(particles_at_frame):
        t = t[t < len(particles_at_frame)]
        warnings.warn(f'You have {len(particles_at_frame)} frames, so the max d_frame you can calculate is {len(particles_at_frame)-1}. max(d_frames) was {max(t)}. I removed down to {max(t)}.')

    t = np.array(t) # (possible) list to ndarray
    num_timesteps = times_at_frame.size

    k_x, k_y, k_bins = get_k_and_bins_for_intermediate_scattering(min_k, max_k, num_k_bins, use_doublesided_k=use_doublesided_k)
    num_k_bins = k_bins.size - 1 # I'm sure this is here for a reason but what is the reason?
    assert np.isfinite(k_x).all()
    assert np.isfinite(k_y).all()

    min_process_size = (k_x.size * k_y.size * particles_at_frame[0, :, 0].size) + (k_x.size * k_y.size * particles_at_frame[0, :, 1].size)
    min_process_ram = min_process_size*particles_at_frame.itemsize

    if not quiet: print(f'RAM to each process: {min_process_size*particles_at_frame.itemsize/1e9:.1f}GB')

    while (total_ram := min_process_ram * cores) > 60e9:
        if cores == 1:
            raise Exception('even with just 1 core, the total RAM was still over 60GB')
        cores = int(cores/2)

    assert total_ram < 60e9

    if not quiet: print(f'min total RAM: {total_ram/1e9:.1f}GB')

    if total_ram > 20e9:
        warnings.warn(f'total RAM usage about {total_ram/1e9:.0f}GB ({cores} cores)')

    F    = np.full((len(t), num_k_bins), np.nan, dtype=np.float32)
    F_unc = np.full((len(t), num_k_bins), np.nan, dtype=np.float32)
    # print(f'F size {common.arraysize(Fs)}')
    k   = np.full((num_k_bins), np.nan, dtype=np.float32)

    F_full     = np.full((len(t), k_x.size, k_y.size), np.nan, dtype=np.float32)
    F_unc_full = np.full((len(t), k_x.size, k_y.size), np.nan, dtype=np.float32)
    k_full     = np.full((len(t), k_x.size, k_y.size), np.nan, dtype=np.float32)

    progress = progressbar(total=len(t)*min(num_timesteps-1, max_time_origins), smoothing=0.03, desc='computing', disable=quiet) # low smoothing makes it more like average speed and less like instantaneous speed

    if cores > 1:
        pool = multiprocessing.Pool(cores)
        if cores > 16:
            warnings.warn(f'using {cores} cores')
        else:
            if not quiet: print(f'using {cores} cores')
    else:
        pool = None
        if not quiet: print('running single threaded')

    if window == 'blackmanharris':
        if not quiet: print('using blackman-harris windowing')
        assert Lx and np.isfinite(Lx), 'when using blackman-harris windowing, you must supply the extent of the data'
        assert Ly and np.isfinite(Ly), 'when using blackman-harris windowing, you must supply the extent of the data'
        window_func = functools.partial(blackman_harris_window, Lx, Ly)
    else:
        window_func = no_window

    # before we do the calculation we get the pairs of frames
    # this is so if there are no frames with a certain dt, we can raise the error before the calculation starts
    # which is a lot less anoying than it failing halfway through
    pairs_at_t = []
    assert times_at_frame.dtype == t.dtype, f"times_at_frame ({times_at_frame.dtype}) and t ({t.dtype}) must be the same dtype "
    for t_i in range(len(t)):
        pairs = get_frames_with_delta_t(times_at_frame, t[t_i])
        assert len(pairs)
        pairs_at_t.append(pairs)
        
    # do the actual calculation
    for t_i in range(len(t)):
        F_, F_unc_, k_, F_unbinned, F_unc_unbinned, k_unbinned = intermediate_scattering_for_dframe(F_type=F_type,
                            max_time_origins=max_time_origins, t=t, particles_at_frame=particles_at_frame,
                            pairs=pairs_at_t[t_i],
                            k_x=k_x, k_y=k_y, k_bins=k_bins, pool=pool, progress=progress, window_func=window_func)
        
        F   [t_i, :] = F_
        F_unc[t_i, :] = F_unc_
        if t_i == 0:
            k[:] = k_ # k is the same from all iterations so we only save it on the first
        F_full    [t_i, :, :] = F_unbinned
        F_unc_full[t_i, :, :] = F_unc_unbinned
        k_full    [t_i, :, :] = k_unbinned

    if cores > 1:
        pool.close()

    assert np.isfinite(k).sum()/k.size > 0.5

    assert np.any(F > 0.001)

    Results = collections.namedtuple('Results', ['F', 'F_unc', 'k', 'F_full', 'F_unc_full', 'k_full', 'k_x', 'k_y', 'd_frames'])
    return Results(F=F, F_unc=F_unc, k=k, F_full=F_full, F_unc_full=F_unc_full, k_full=k_full, k_x=k_y, k_y=k_y, d_frames=t)

def intermediate_scattering_for_dframe(F_type, max_time_origins, t, particles_at_frame, k_x, k_y, k_bins, pool, progress, window_func, pairs):
    assert particles_at_frame.dtype == np.float32

    use_every_nth_pair = int(np.ceil(pairs.shape[0] / max_time_origins))

    pairs_of_particles = []

    for [frame1, frame2] in pairs[::use_every_nth_pair, :]:
        pairs_of_particles.append((particles_at_frame[frame1, :, :], particles_at_frame[frame2, :, :]))
    
    num_used_time_origins = len(pairs_of_particles)
    mean_num_particles = np.count_nonzero(np.isfinite(particles_at_frame)) / particles_at_frame.shape[0] / 2 # div 2 for x and y

    F = np.full((num_used_time_origins, k_bins.size-1), np.nan, dtype=np.float32)
    k = np.full((k_bins.size-1),                        np.nan, dtype=np.float32) # +1 b/c we get the left and right of the final bin
    F_full     = np.full((num_used_time_origins, k_x.size, k_y.size), np.nan, dtype=np.float32)
    k_full     = np.full((k_x.size, k_y.size),                        np.nan, dtype=np.float32)
                    
    if F_type == 'F_s':
        func = self_intermediate_scattering_internal
    else:
        func = intermediate_scattering_internal
        # func = intermediate_scattering_internal_incremental

    bound = functools.partial(intermediate_scattering_preprocess_run_postprocess,
                                k_x, k_y, k_bins, func, window_func)

    results = []
    if pool:
        tasks = pool.imap(bound, pairs_of_particles, chunksize=1)
        for result in tasks:
            results.append(result)
            progress.update()
    else:
        for particleset in pairs_of_particles:
            results.append(bound(particleset))
            progress.update()

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

    assert np.isfinite(k).all()

    # need nanmean because binned_statistic will return nan if the bin is empty
    return np.nanmean(F, axis=0), np.nanstd(F, axis=0)/np.sqrt(num_used_time_origins*mean_num_particles), k, np.nanmean(F_full, axis=0), np.nanstd(F_full, axis=0)/np.sqrt(num_used_time_origins), k_full


def intermediate_scattering_preprocess_run_postprocess(k_x, k_y, k_bins, func, window_func, particles):
    # there can be nan in particles_t0/1 if we're calculating F b/c it's a numpy array padded with nan at the top

    k_unbinned, F_unbinned = func(particles[0], particles[1], k_x, k_y, window_func)
    # func is probably intermediate_scattering_internal
    
    assert np.isnan(F_unbinned).sum() <= 1, f'F was {np.isnan(F_unbinned).sum()/F_unbinned.size*100:.0f}% NaN'
    # one nan point is allowed at k=(0, 0)
    
    k_binned, F_binned = postprocess_scattering(k_unbinned, F_unbinned, k_bins)
    return k_binned, F_binned, k_unbinned, F_unbinned

def postprocess_scattering(k, F, k_bins):
    # this is where we do the angular average over k space
    # k and F are shape (num k points x) * (num k points y)

    F_flat = F.flatten()
    k_flat = k.flatten()

    # there should be just one nan value at k_x == k_y == 0
    nans = np.isnan(F_flat)
    assert nans.sum() == 1 # apparently there can be nan in particles_t0/1 if we're calculating F b/c it's a numpy array padded with nan at the top

    assert np.isfinite(k_flat).all()

    # we should probably filter out points at k > k_bins[max] 
    
    assert np.all(k_flat[~nans] >= k_bins[0] ), f'k_flat[~nans].min() = {k_flat[~nans].min()}, k_bins[0] = {k_bins[0]}'
    
    F_binned, _, _ = scipy.stats.binned_statistic(k_flat[~nans], F_flat[~nans], 'mean', bins=k_bins)
    k_binned, _, _ = scipy.stats.binned_statistic(k_flat[~nans], k_flat[~nans], 'mean', bins=k_bins)

    assert np.isfinite(k_binned).sum() / k_binned.size > 0.5, f'k_binned finite: {np.isfinite(k_binned).sum()/k_binned.size}'
    
    # ^^ we used to use the middle of the bin, but this will be slightly skewed for small k, so we do a proper average
    # binned statistic returns NaN if the bin is empty
    # this is okay for F, but it's annoying if k includes nans
    # so we make use the bin middle for those bins that are empty to satisfy downstream code
    bin_mids = (k_bins[1:] + k_bins[:-1]) / 2
    k_binned[np.isnan(k_binned)] = bin_mids[np.isnan(k_binned)]
    assert np.isfinite(k_binned).all()

    assert np.isnan(F_binned).sum() < F_binned.size

    return k_binned, F_binned

def quantise(values, min_k):
    # return values with each element replaced by the closest multiple of min_k
    v = values / min_k
    v = np.round(v)
    quantised = v * min_k
    quantised = np.unique(quantised)
    return quantised

def get_k_and_bins_for_intermediate_scattering(min_k, max_k, num_k_bins, use_doublesided_k=False):
    # only allowed k values are multiples of min_k
    # but we want log-spaced values
    k_oneside_x = np.logspace(np.log10(min_k[0]), np.log10(max_k), num_k_bins, dtype=np.float32)
    # so we take log-spaced values and move them to the nearest multiple of min_k
    k_oneside_x = quantise(k_oneside_x, min_k[0])
    
    k_oneside_y = np.logspace(np.log10(min_k[1]), np.log10(max_k), num_k_bins, dtype=np.float32)
    k_oneside_y = quantise(k_oneside_y, min_k[1])

    k_x = np.concatenate([-k_oneside_x[::-1], (0,), k_oneside_x], dtype=np.float32)

    if use_doublesided_k:
        k_y = np.concatenate([-k_oneside_y[::-1], (0,), k_oneside_y], dtype=np.float32)

    else:
        k_y = np.concatenate([(0,), k_oneside_y], dtype=np.float32)

    bin_edges = np.concatenate([(0,), k_oneside_x], dtype=np.float32)

    assert np.unique(bin_edges).size == bin_edges.size, 'duplicate values were found in bin_edges'
    assert np.all(k_oneside_x >= bin_edges[0] ), f'k_oneside_x.min() = {k_oneside_x.min()}, bin_edges[0] = {bin_edges[0]}'

    assert np.isnan(k_x).sum() == 0
    assert np.isnan(k_y).sum() == 0

    # there's a small problem here as we double count the points on k_x = 0 when on not doublesided mode

    assert k_x.dtype == np.float32, f'k_x.dtype = {k_x.dtype}'

    return k_x, k_y, bin_edges

def blackman_harris_window(Lx, Ly, x, y):
    # Giavazzi, F., Edera, P., Lu, P.J. et al. Image windowing mitigates edge effects in Differential Dynamic Microscopy
    # https://link.springer.com/article/10.1140/epje/i2017-11587-3
    # eq 10
    # assert np.all(x <= Lx)
    # assert np.all(y <= Ly)

    a = [0.3635819, 0.4891775, 0.1365995, 0.0106411]
    sum_internal = lambda j, coord, L: (-1)**j * a[j] * np.cos(2*np.pi * j * coord / L )
    W_BN_x = np.sum(np.array([sum_internal(j, x, Lx) for j in [0, 1, 2, 3]]), axis=0)
    W_BN_y = np.sum(np.array([sum_internal(j, y, Ly) for j in [0, 1, 2, 3]]), axis=0)
    return W_BN_x * W_BN_y

def no_window(x, y):
    return np.ones_like(x)

def intermediate_scattering_internal(particles_t0, particles_t1, k_x, k_y, window_func):

    assert k_x.dtype == np.float32, f'k_x.dtype = {k_x.dtype}'
    assert particles_t0.dtype == np.float32, f'particles_t0.dtype = {particles_t0.dtype}'
    # Thorneywork et al 2018 eq (27))
    
    # there are nans b/c we padded a numpy array with them
    t0_nans = np.any(np.isnan(particles_t0), axis=1)
    t1_nans = np.any(np.isnan(particles_t1), axis=1)
    particles_t0 = particles_t0[~t0_nans, :]
    particles_t1 = particles_t1[~t1_nans, :]
    
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

    k_dot_r_mu = k_x * x_mu + k_y * y_mu # this used to by dtype=f64 but idk why
    k_dot_r_nu = k_x * x_nu + k_y * y_nu

    # set the k=(0, 0) point to nan
    k_dot_r_mu[:, :, x_zero_index, y_zero_index] = np.nan
    k_dot_r_nu[:, :, x_zero_index, y_zero_index] = np.nan

    mu_weights = window_func(x_mu, y_mu)
    nu_weights = window_func(x_nu, y_nu)

    cos_term1 = (np.cos(k_dot_r_mu) * mu_weights).sum(axis=0) # sum over mu
    cos_term2 = (np.cos(k_dot_r_nu) * nu_weights).sum(axis=1) # sum over nu
    cos_accum = cos_term1 * cos_term2
    del cos_term1, cos_term2
    
    sin_term1 = (np.sin(k_dot_r_mu) * mu_weights).sum(axis=0)
    sin_term2 = (np.sin(k_dot_r_nu) * nu_weights).sum(axis=1)
    sin_accum = sin_term1 * sin_term2
    assert k_dot_r_mu.dtype == np.float32
    del sin_term1, sin_term2
    del k_dot_r_mu, k_dot_r_nu
    
    # num_particles = (particles_t0.shape[0] + particles_t1.shape[0]) / 2
    num_particles = (mu_weights.sum() + nu_weights.sum()) / 2

    if num_particles == 0:
        warnings.warn('found no particles in either timestep')
        f = np.zeros_like(cos_accum)
    else:
        f = 1/num_particles * ( cos_accum + sin_accum )
    # del cos_accum, sin_accum # probably unneeded

    f = np.squeeze(f) # idk why by shape is currently (1, 132, 156)
    assert np.isnan(f[x_zero_index, y_zero_index]), f'f[x_zero_index, y_zero_index] = {f[x_zero_index, y_zero_index]}'

    assert np.isfinite(k).all()

    return k, f

"""
def intermediate_scattering_internal_incremental(particles_t0, particles_t1, k_x, k_y):
    # Thorneywork et al 2018 eq (27))
    # print('starting calc')
    
    # there are nans b/c we padded a numpy array with them
    t0_nans = np.any(np.isnan(particles_t0), axis=1)
    t1_nans = np.any(np.isnan(particles_t1), axis=1)
    particles_t0 = particles_t0[~t0_nans, :]
    particles_t1 = particles_t1[~t1_nans, :]
    
    particle_t0_x = particles_t0[:, 0]
    particle_t0_y = particles_t0[:, 1]
    particle_t1_x = particles_t1[:, 0]
    particle_t1_y = particles_t1[:, 1]

    # dimensions are
    # nu x kx x ky
    # x_mu = particle_t0_x[:, np.newaxis, np.newaxis, np.newaxis]
    # y_mu = particle_t0_y[:, np.newaxis, np.newaxis, np.newaxis]
    x_nu = particle_t1_x[:, np.newaxis, np.newaxis]
    y_nu = particle_t1_y[:, np.newaxis, np.newaxis]
    # print('x dtype', x_mu.dtype)

    k_x_ = k_x[:, np.newaxis]
    k_y_ = k_y[np.newaxis, :]
    k = np.sqrt(k_x_**2 + k_y_**2)

    # we gonna set the k=(0, 0) point to nan later, for now we save where it is
    assert np.sum(k_x == 0) == 1, f'np.sum(k_x == 0) = {np.sum(k_x == 0)}'
    assert np.sum(k_y == 0) == 1, f'np.sum(k_y == 0) = {np.sum(k_y == 0)}'
    k_x_zero_index = np.argmax(k_x == 0)
    k_y_zero_index = np.argmax(k_y == 0)
    assert k_x[k_x_zero_index] == 0
    assert k_y[k_y_zero_index] == 0

    all_f = np.full((particles_t0.shape[0], k_x.size, k_y.size), np.nan, dtype=np.float32)
    print(f'all_f {all_f.nbytes/1e9:.1f}GB')

    k_x = k_x[np.newaxis, :, np.newaxis]
    k_y = k_y[np.newaxis, np.newaxis, :]

    assert k_x.dtype == np.float32, f'k_x.dtype = {k_x.dtype}'
    assert particles_t0.dtype == np.float32, f'particles_t0.dtype = {particles_t0.dtype}'
    
    for mu in range(particles_t1.shape[0]):
        x_mu = particle_t0_x[mu]
        y_mu = particle_t0_y[mu]

        k_dot_r_mu = k_x * x_mu  +  k_y * y_mu # this used to by dtype=f64 but idk why
        k_dot_r_nu = k_x * x_nu  +  k_y * y_nu

        # set the k=(0, 0) point to nan
        k_dot_r_mu[:, k_x_zero_index, k_y_zero_index] = np.nan
        k_dot_r_nu[:, k_x_zero_index, k_y_zero_index] = np.nan

        cos_term1 = np.cos(k_dot_r_mu) # sum over mu
        cos_term2 = np.cos(k_dot_r_nu).sum(axis=0) # sum over nu
        cos_accum = cos_term1 * cos_term2
        del cos_term1, cos_term2
        
        sin_term1 = np.sin(k_dot_r_mu)
        sin_term2 = np.sin(k_dot_r_nu).sum(axis=0)
        sin_accum = sin_term1 * sin_term2
        # print(f'k_dot_r_mu+k_dot_r_nu {k_dot_r_mu.nbytes/1e9+k_dot_r_nu.nbytes/1e9:.1f}GB')
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
        assert np.isnan(f[k_x_zero_index, k_y_zero_index])

        all_f[mu, :, :] = f

    all_f = all_f.sum(axis=0) # sum over mu

    # print('finished calc')
    return k, all_f"
"""

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