import numpy as np
import numba
import common
import tqdm
import time, functools
import multiprocessing
import warnings

###########
# there is a lot of stuff in here that needs tidying
# the entry points is calc_incremental
# unless the timesteps in your data are unevenly spaced, in which case it is calc_mixt_new
# calc_incremental_xyz will return the MSD of each dimension separately
###########



# @numba.njit
def autocorrFFT(x):
    N = len(x)
    F = np.fft.fft(x, n=2*N)  # 2*N because of zero-padding
    PSD = F * np.conjugate(F)
    res = np.fft.ifft(PSD)
    res = (res[:N]).real  # now we have the autocorrelation in convention B
    n = N * np.ones(N) - np.arange(0, N)  # divide res(m) by (N-m)
    return res / n  # this is the autocorrelation in convention A

@numba.njit(fastmath=True)
def msd_fft1d(r):
    N = len(r)
    D = np.square(r)
    D = np.append(D, 0)
    with numba.objmode(S2='float64[:]'):
        S2 = autocorrFFT(r) # we have to run in objmode cause numba does not support fft
    Q = 2 * D.sum()
    S1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m-1] - D[N-m]
        S1[m] = Q / (N-m)
    return S1 - 2 * S2

@numba.njit(parallel=True, fastmath=True)
def msd_matrix(matrix):
    # calculates the MSDs of the rows of the provided matrix
    Nrows, Ncols = matrix.shape
    MSDs = np.zeros((Nrows,Ncols))
    for i in range(Nrows):
        #print(100.0 * ((1.0 * i) / (1.0 * N)), "percent done with MSD calc")
        MSD = msd_fft1d(matrix[i, :])
        MSDs[i,:] = MSD
    return MSDs


def reshape(particles):    
    # reformat to be (num particles) x (num timesteps) x (x, y)
    assert particles.shape[1] == 4, 'you need to use linked data'
    
    num_particles = int(particles[:, 3].max()) + 1
    num_timesteps = int(particles[:, 2].max()) + 1

    assert particles[:, 3].min() == 0, f'smallest ID was {particles[:, 3].min()}'
    assert particles[:, 2].min() == 0

    data = np.full((num_particles, num_timesteps, 2), np.nan)

    for row_index in tqdm.trange(particles.shape[0], desc='reshaping'):
        x, y, t, num = particles[row_index]
        data[int(num)-0, int(t)-0, :] = x, y # -0 cause they are 0-based

    return data

def calc_internal(data):
    print('nanfrac before', common.nanfrac(data))

    i = 0
    ps = np.full_like(data, np.nan)
    for particle in range(data.shape[0]):
        # print(common.nanfrac(data[particle, :, :]))
        if common.nanfrac(data[particle, :, :]) == 0:
            ps[i, :, :] = data[particle, :, :]
            i += 1

    hist = [common.nanfrac(data[p, :, :]) for p in range(data.shape[0])]
    print('nanfrac ')
    common.term_hist(hist)

    ps = ps[:i, :, :]
    data = ps

    print('calculating MSDs')
    x_MSDs = msd_matrix(data[:, :, 0]) # should probably be msd_fft1d?
    y_MSDs = msd_matrix(data[:, :, 1])
    MSDs = x_MSDs + y_MSDs
    print('nanfrac after', common.nanfrac(MSDs))

    mean, std = np.nanmean(MSDs, axis=0), np.nanstd(MSDs, axis=0)

    assert common.nanfrac(mean) < 0.1, f'MSDs nanfrac was {common.nanfrac(mean)}'

    return mean, std

def calc(particles):
    data = reshape(particles)
    # data should be of shape (num particles) x (num timesteps) x (2)

    return calc_internal(data)

def calc_individuals(particles):
    data = reshape(particles)
    # data should be of shape (num particles) x (num timesteps) x (2)
    
    x_MSDs = msd_matrix(data[:, :, 0])
    y_MSDs = msd_matrix(data[:, :, 1])
    MSDs = x_MSDs + y_MSDs

    return MSDs

# def calc_centre_of_mass(data, groupsize):

#     np.random.shuffle(data) # shuffles along first axes (num particles)

#     groups = np.array_split(data, np.ceil(data.shape[0]/groupsize), axis=0) # ceil needed otherwise the groups will be 1 too big because of roundind down
    
#     centre_of_masses = np.array([np.nanmean(group, axis=0) for group in groups])
#     print('need to consider that many of your group might be nan')
#     return calc_internal(centre_of_masses)


@numba.njit(parallel=True)
def numba_isin(needles, haystack):
    ans = np.full(needles.shape, False)
    for i in range(len(needles)):
        for j in range(len(haystack)):
            if needles[i] == haystack[j]:
                ans[i] = True
                # break # "prange or pndindex loop will not be executed in parallel due to there being more than one entry to or exit from the loop (e.g., an assertion)."
    return ans

@numba.njit(parallel=True)
def calc_centre_of_mass_onepoint_incremental_numba(particles, groupsize, num_time_origins):
    num_timesteps = particles[:, 2].max()
    
    num_time_origins = int(min(num_time_origins, num_timesteps))
    time_origins = [int(i) for i in np.linspace(0, num_timesteps-2, num_time_origins)]
    
    displacements = np.full(num_time_origins, np.nan)

    for time_origin_index in range(num_time_origins):
        t0 = time_origins[time_origin_index]
        t1 = t0 + 1

        # find the rows at the relevent time steps
        particles_t0 = particles[particles[:, 2] == t0]
        particles_t1 = particles[particles[:, 2] == t1]

        all_particle_ids_at_t0 = particles_t0[:, 3]
        all_particle_ids_at_t1 = particles_t1[:, 3]
        # find the particles who exist at both of these time steps
        particle_ids_at_these_timesteps = np.intersect1d(all_particle_ids_at_t0, all_particle_ids_at_t1)

        # find the rows containing these particles at the relevent time step
        particles_t0 = particles_t0[np.isin(particles_t0[:, 3], particle_ids_at_these_timesteps), :]
        particles_t1 = particles_t1[np.isin(particles_t1[:, 3], particle_ids_at_these_timesteps), :]

        # sort by particle ID
        particles_t0 = particles_t0[particles_t0[:, 3].argsort(), :]
        particles_t1 = particles_t1[particles_t1[:, 3].argsort(), :]

        # assert np.all(np.isin(particles_t0[:, 3], all_particle_ids_at_t0)) # assertions commented after development for performance
        # assert np.all(np.isin(particles_t1[:, 3], all_particle_ids_at_t1))
        # assert np.all(particles_t0[:, 3] == particles_t1[:, 3])
        data_t0 = particles_t0[:, 0:2]
        data_t1 = particles_t1[:, 0:2]
        
        shuffle = np.random.permutation(data_t0.shape[0])
        data_t0 = data_t0[shuffle, :]
        data_t1 = data_t1[shuffle, :]

        num_groups = particle_ids_at_these_timesteps.size // groupsize
        displacements_at_time = np.full(num_groups, np.nan)
            
        for group_i in numba.prange(num_groups):
            data_this_group_t0 = data_t0[group_i*groupsize:(group_i+1)*groupsize, :]
            data_this_group_t1 = data_t1[group_i*groupsize:(group_i+1)*groupsize, :]

            # assert data_this_group_t0.shape[0] == groupsize

            # COM_start = data_this_group_t0[:, :].mean(axis=0)
            # COM_end   = data_this_group_t1[:, :].mean(axis=0)
            COM_start = common.numba_mean_2d_axis0(data_this_group_t0)
            COM_end   = common.numba_mean_2d_axis0(data_this_group_t1)


            displacement = np.sum((COM_end - COM_start)**2)
            displacements_at_time[group_i] = displacement
        
        displacements[time_origin_index] = displacements_at_time.mean()

    return displacements.mean(), displacements.std()/np.sqrt(displacements.size)

def calc_centre_of_mass_onepoint_incremental(particles, groupsizes, num_time_origins):
    num_timesteps = particles[:, 2].max()
    
    num_time_origins = int(min(num_time_origins, num_timesteps))
    time_origins = [int(i) for i in np.linspace(0, num_timesteps-2, num_time_origins)]
    
    displacements = np.full((len(groupsizes), num_time_origins), np.nan)

    for time_origin_index in tqdm.trange(num_time_origins):
        t0 = time_origins[time_origin_index]
        t1 = t0 + 1
        
        # find the rows at the relevent time steps
        particles_t0 = particles[particles[:, 2] == t0]
        particles_t1 = particles[particles[:, 2] == t1]

        all_particle_ids_at_t0 = particles_t0[:, 3]
        all_particle_ids_at_t1 = particles_t1[:, 3]
        # find the particles who exist at both of these time steps
        particle_ids_at_these_timesteps = np.intersect1d(all_particle_ids_at_t0, all_particle_ids_at_t1)

        # find the rows containing these particles at the relevent time step
        particles_t0 = particles_t0[np.isin(particles_t0[:, 3], particle_ids_at_these_timesteps), :]
        particles_t1 = particles_t1[np.isin(particles_t1[:, 3], particle_ids_at_these_timesteps), :]

        # sort by particle ID
        particles_t0 = particles_t0[particles_t0[:, 3].argsort(), :]
        particles_t1 = particles_t1[particles_t1[:, 3].argsort(), :]

        # assert np.all(np.isin(particles_t0[:, 3], all_particle_ids_at_t0)) # assertions commented after development for performance
        # assert np.all(np.isin(particles_t1[:, 3], all_particle_ids_at_t1))
        # assert np.all(particles_t0[:, 3] == particles_t1[:, 3])
        data_t0 = particles_t0[:, 0:2]
        data_t1 = particles_t1[:, 0:2]
        
        shuffle = np.random.permutation(data_t0.shape[0])
        data_t0 = data_t0[shuffle, :]
        data_t1 = data_t1[shuffle, :]

        for groupsize_index in range(len(groupsizes)):
            groupsize = groupsizes[groupsize_index]
            num_groups = particle_ids_at_these_timesteps.size // groupsize
            displacements_at_time = np.full(num_groups, np.nan)
                
            for group_i in range(num_groups):
                data_this_group_t0 = data_t0[group_i*groupsize:(group_i+1)*groupsize, :]
                data_this_group_t1 = data_t1[group_i*groupsize:(group_i+1)*groupsize, :]

                # assert data_this_group_t0.shape[0] == groupsize

                COM_start = data_this_group_t0[:, :].mean(axis=0)
                COM_end   = data_this_group_t1[:, :].mean(axis=0)
                # COM_start = common.numba_mean_2d_axis0(data_this_group_t0)
                # COM_end   = common.numba_mean_2d_axis0(data_this_group_t1)


                displacement = np.sum((COM_end - COM_start)**2)
                displacements_at_time[group_i] = displacement
            
            displacements[groupsize_index, time_origin_index] = displacements_at_time.mean()

    return displacements.mean(axis=1), displacements.std(axis=1)/np.sqrt(displacements.shape[1])
    
# def calc_centre_of_mass_incremental(particles, groupsizes, d_frames, num_time_origins):
#     num_timesteps = particles[:, 2].max()
    
#     # num_time_origins = int(min(num_time_origins, num_timesteps))
    
#     displacements     = np.full((len(groupsizes), len(d_frames), num_time_origins), np.nan)
#     num_displacements = np.full((len(groupsizes), len(d_frames)), 0)

#     progress = tqdm.tqdm(total=len(d_frames) * num_time_origins * len(groupsizes))
#     for dframe_index, d_frame in enumerate(d_frames):
#         time_origins = [int(i) for i in np.linspace(0, num_timesteps-d_frame-1, num_time_origins)]
    
#         for time_origin_index in range(num_time_origins):
#             t0 = time_origins[time_origin_index]
#             t1 = t0 + d_frame
            
#             # find the rows at the relevent time steps
#             particles_t0 = particles[particles[:, 2] == t0]
#             particles_t1 = particles[particles[:, 2] == t1]

#             all_particle_ids_at_t0 = particles_t0[:, 3]
#             all_particle_ids_at_t1 = particles_t1[:, 3]
#             # find the particles who exist at both of these time steps
#             particle_ids_at_these_timesteps = np.intersect1d(all_particle_ids_at_t0, all_particle_ids_at_t1)
#             if particle_ids_at_these_timesteps.size == 0:
#                 continue

#             # find the rows containing these particles at the relevent time step
#             particles_t0 = particles_t0[np.isin(particles_t0[:, 3], particle_ids_at_these_timesteps), :]
#             particles_t1 = particles_t1[np.isin(particles_t1[:, 3], particle_ids_at_these_timesteps), :]

#             # sort by particle ID
#             particles_t0 = particles_t0[particles_t0[:, 3].argsort(), :]
#             particles_t1 = particles_t1[particles_t1[:, 3].argsort(), :]

#             # assert np.all(np.isin(particles_t0[:, 3], all_particle_ids_at_t0)) # assertions commented after development for performance
#             # assert np.all(np.isin(particles_t1[:, 3], all_particle_ids_at_t1))
#             # assert np.all(particles_t0[:, 3] == particles_t1[:, 3])
#             data_t0 = particles_t0[:, 0:2]
#             data_t1 = particles_t1[:, 0:2]
            
#             shuffle = np.random.permutation(data_t0.shape[0])
#             data_t0 = data_t0[shuffle, :]
#             data_t1 = data_t1[shuffle, :]

#             for groupsize_index in range(len(groupsizes)):
#                 groupsize = groupsizes[groupsize_index]
#                 if particle_ids_at_these_timesteps.size < groupsize:
#                     continue

#                 num_groups = particle_ids_at_these_timesteps.size // groupsize
#                 assert num_groups > 0
#                 displacements_at_time = np.full(num_groups, np.nan)
                    
#                 for group_i in range(num_groups):
#                     data_this_group_t0 = data_t0[group_i*groupsize:(group_i+1)*groupsize, :]
#                     data_this_group_t1 = data_t1[group_i*groupsize:(group_i+1)*groupsize, :]

#                     # assert data_this_group_t0.shape[0] == groupsize

#                     COM_start = data_this_group_t0[:, :].mean(axis=0)
#                     COM_end   = data_this_group_t1[:, :].mean(axis=0)
#                     # COM_start = common.numba_mean_2d_axis0(data_this_group_t0)
#                     # COM_end   = common.numba_mean_2d_axis0(data_this_group_t1)


#                     displacement = np.sum((COM_end - COM_start)**2)
#                     assert not np.isnan(displacement)
#                     displacements_at_time[group_i] = displacement

#                 progress.update()
                
#                 displacements[groupsize_index, dframe_index, time_origin_index] = displacements_at_time.mean()
#                 num_displacements[groupsize_index, dframe_index] += 1
#     progress.close()
#     print(num_displacements)

#     # we currently overestimate num_displacements, because they're not all independent if the time origins overlap

#     return np.nanmean(displacements, axis=2), np.nanstd(displacements, axis=2)/np.sqrt(num_displacements)

def calc_centre_of_mass_proximity(particles, num_timesteps, window_size_x, window_size_y, num_time_origins, box_sizes):
    num_timesteps = int(particles[:, 2].max()) + 1

    time_origins = [int(t) for t in np.linspace(0, num_timesteps, num_time_origins+1)[:-1]]

    particles_divided_by_id = divide_particles_by_id(particles, num_timesteps)

    # print('sorting')
    # particles = particles[particles[:, 2].argsort()]
    # print('sorted')

    msds = np.full((len(box_sizes), len(time_origins), num_timesteps), np.nan)
    num_used_particles = np.full((len(box_sizes), len(time_origins)), np.nan)

    progress = tqdm.tqdm(total=len(time_origins)*len(box_sizes))

    for time_origin_index, time_origin in enumerate(time_origins):
        particles_at_frame = particles[particles[:, 2] == time_origin]

        for box_size_index, box_size in enumerate(box_sizes):
            max_num_boxes_x = int(window_size_x // box_size)
            max_num_boxes_y = int(window_size_y // box_size)
            box_x_positions = np.array(range(0, max_num_boxes_x)) * box_size
            box_y_positions = np.array(range(0, max_num_boxes_y)) * box_size
            while box_x_positions.size > 10:
                box_x_positions = box_x_positions[::2]
            while box_y_positions.size > 10:
                box_y_positions = box_y_positions[::2]
            num_boxes_x = box_x_positions.size
            num_boxes_y = box_y_positions.size

            msds_this_boxsize = np.full((num_boxes_x, num_boxes_y, num_timesteps), np.nan)
            num_this_boxsize  = np.full((num_boxes_x, num_boxes_y), np.nan)

            for box_x_index in range(num_boxes_x):
                box_x_min = box_x_positions[box_x_index]
                box_x_max = box_x_min + box_size
                for box_y_index in range(num_boxes_y):
                    # print('a box')
                    box_y_min = box_y_positions[box_y_index]
                    box_y_max = box_y_min + box_size

                    # find particles in box
                    in_x = (box_x_min <= particles_at_frame[:, 0]) & (particles_at_frame[:, 0] < box_x_max)
                    in_y = (box_y_min <= particles_at_frame[:, 1]) & (particles_at_frame[:, 1] < box_y_max)
                    particle_ids_in_box = particles_at_frame[in_x & in_y, 3]
                    # particles_in_box = particles_at_frame[
                    # print(particle_ids_in_box.size)
                    if particle_ids_in_box.size == 0:
                        continue # no! probably we should add zeros!!

                    particles_reshaped = np.full((particle_ids_in_box.size, num_timesteps-time_origin, 4), np.nan)

                    for particle_index, particle_id in enumerate(particle_ids_in_box):
                        rows_this_particle = particles_divided_by_id[int(particle_id)]
                        start_row = np.nonzero(rows_this_particle[:, 2] == time_origin)[0][0]
                        particles_reshaped[particle_index, :rows_this_particle.shape[0]-start_row, :] = rows_this_particle[start_row:, :]

                    # find time when one of the particles dissapears
                    for end_time in range(particles_reshaped.shape[1]):
                        timesteps = particles_reshaped[:, end_time, 2]

                        if np.any(np.isnan(timesteps)):
                            break
                        
                        if np.any(timesteps != timesteps[0]):
                            break

                        if end_time > 0:
                            if timesteps[0] != last_timestep + 1:
                                break

                        last_timestep = timesteps[0]

                    # get the centre of mass
                    centre_of_mass = particles_reshaped[:, :end_time, [0, 1]].mean(axis=0)
                    msd = msd_fft1d(centre_of_mass[:, 0]) + msd_fft1d(centre_of_mass[:, 1])
                    
                    msds_this_boxsize[box_x_index, box_y_index, :msd.size] = msd
                    num_this_boxsize [box_x_index, box_y_index]            = particle_ids_in_box.size

            msds[box_size_index, time_origin_index, :] = np.nanmean(msds_this_boxsize, axis=(0, 1)) # mean across boxes in x and y
            num_used_particles[box_size_index, time_origin_index] = np.nanmean(num_this_boxsize)

            progress.update()

    progress.close()
    return np.nanmean(msds, axis=1), np.nanmean(num_used_particles, axis=1)

def find_particle_ids_at_frame(target_frame, particles):
    return np.unique(particles[particles[:, 2] == target_frame, 3])

def find_particles_at_all_frames(particles, num_timesteps):
    # first we divide up the data into frames, which allows later code to be more efficient
    num_particles_at_frame = np.bincount(particles[:, 2].astype('int'))
    max_particles_at_frame = num_particles_at_frame.max()
    particles_at_frame = np.full((num_timesteps, max_particles_at_frame, 4), np.nan)
    used_slots         = np.full((num_timesteps), 0)

    for row_index in tqdm.trange(particles.shape[0], desc='finding particles at frame'):
        row = particles[row_index, :]
        row_timestep = int(row[2])
        if not np.isin(row[3], particles_at_frame[row_timestep]):
            particles_at_frame[row_timestep, used_slots[row_timestep]] = row[3]
            used_slots[row_timestep] += 1

    return particles_at_frame

def divide_particles_by_id(particles, num_timesteps):
    # first we divide up the data by particle id, which allows later code to be more efficient
    particles_divided = [] # if we used a numpy array this would be reshaping as in the past, which is a no-go
    num_particles = particles[:, 3].max() + 1
    
    # need the data sorted by ID
    print('sorting')
    particles = particles[particles[:, 3].argsort()]
    print('sorted')

    start_index = 0
    current_id = 0
    progress = tqdm.tqdm(total=num_particles, desc='dividing')
    while start_index < particles.shape[0]:
        current_id = particles[start_index, 3]
        end_index = start_index
        
        assert len(particles_divided) == current_id

        # find the index of the last row that has this particle ID
        while end_index < particles.shape[0] and particles[end_index, 3] == current_id:
            end_index += 1

        particles_to_append = particles[start_index:end_index, :]
        particles_to_append = particles_to_append[particles_to_append[:, 2].argsort()]
        
        particles_divided.append(particles_to_append)

        start_index = end_index
        progress.update()
    progress.close()
    return particles_divided

def divide_particles_by_frame(particles, num_timesteps):
    # first we divide up the data into frames, which allows later code to be more efficient
    num_particles_at_frame = np.bincount(particles[:, 2].astype('int'))
    max_particles_at_frame = num_particles_at_frame.max()
    particles_divided = np.full((num_timesteps, max_particles_at_frame, 4), np.nan)

    # slow but easy version:
    # for timestep in tqdm.trange(num_timesteps, desc='dividing'):
    #     particles_at_timestep = particles[particles[:, 2] == timestep]
    #     particles_divided[timestep, 0:particles_at_timestep.shape[0], :] = particles_at_timestep
    
    # fast version:
    # need the data sorted by time
    print('sorting')
    particles = particles[particles[:, 2].argsort()]
    print('sorted')

    start_index = 0
    current_time = 0
    progress = tqdm.tqdm(total=num_timesteps, desc='dividing')
    while start_index < particles.shape[0]:
        current_time = particles[start_index, 2]
        end_index = start_index

        # find the index of the last row that has this particle ID
        while end_index < particles.shape[0] and particles[end_index, 2] == current_time:
            end_index += 1
        
        particles_divided[int(current_time), 0:end_index-start_index, :] = particles[start_index:end_index, :]

        start_index = end_index
        progress.update()
    progress.close()
    return particles_divided

def calc_centre_of_mass_incremental_numba(particles, groupsizes, d_frames, num_time_origins):
    num_timesteps = int(particles[:, 2].max()) + 1
    
    # num_time_origins = int(min(num_time_origins, num_timesteps))

    # particles = np.array([
    #     [1, 1, 0, 0],
    #     [2, 2, 0, 1],
    #     [3, 3, 1, 0],
    #     [4, 4, 1, 1],
    #     [5, 5, 1, 1],
    #     [6, 6, 2, 0],
    #     [7, 7, 2, 1],
    # ])
    # print('p12', particles[1, 2])
    # print(divide_particles(particles, 3))
    # sys.exit()
    
    displacements     = np.full((len(groupsizes), len(d_frames), num_time_origins), np.nan)
    num_displacements = np.full((len(groupsizes), len(d_frames)), 0)


    particles_divided = divide_particles_by_frame(particles, num_timesteps)

    # now we do the computation
    progress = tqdm.tqdm(total=len(d_frames) * num_time_origins, desc='computing')
    for dframe_index, d_frame in enumerate(d_frames):
        time_origins = [int(i) for i in np.linspace(0, num_timesteps-d_frame-1, num_time_origins)]
    
        for time_origin_index in range(num_time_origins):
            t0 = time_origins[time_origin_index]
            t1 = t0 + d_frame
            
            # find the rows at the relevent time steps
            particles_t0 = particles_divided[t0]
            particles_t1 = particles_divided[t1]

            all_particle_ids_at_t0 = particles_t0[:, 3]
            all_particle_ids_at_t1 = particles_t1[:, 3]
            # find the particles who exist at both of these time steps
            particle_ids_at_these_timesteps = np.intersect1d(all_particle_ids_at_t0, all_particle_ids_at_t1, assume_unique=True) # assume_unique tells numpy that each array has no duplicate values
            
            if particle_ids_at_these_timesteps.size == 0:
                continue

            # find the rows containing these particles at the relevent time step
            particles_t0 = particles_t0[np.isin(particles_t0[:, 3], particle_ids_at_these_timesteps), :]
            particles_t1 = particles_t1[np.isin(particles_t1[:, 3], particle_ids_at_these_timesteps), :]

            displacements_at_time_origin, num_displacements_at_time_origin = calc_centre_of_mass_incremental_numba_internal(
                particles_t0, particles_t1, groupsizes, particle_ids_at_these_timesteps.size
            )
            displacements[:, dframe_index, time_origin_index] = displacements_at_time_origin
            num_displacements[:, dframe_index] += num_displacements_at_time_origin
            
            progress.update()

    progress.close()

    return np.nanmean(displacements, axis=2), np.nanstd(displacements, axis=2)/np.sqrt(num_displacements)

@numba.njit(parallel=True)
# @numba.njit()
def calc_centre_of_mass_incremental_numba_internal(particles_t0, particles_t1, groupsizes, particle_ids_at_these_timesteps_size):
    # sort by particle ID
    particles_t0 = particles_t0[particles_t0[:, 3].argsort(), :]
    particles_t1 = particles_t1[particles_t1[:, 3].argsort(), :]

    # assert np.all(np.isin(particles_t0[:, 3], all_particle_ids_at_t0)) # assertions commented after development for performance
    # assert np.all(np.isin(particles_t1[:, 3], all_particle_ids_at_t1))
    # assert np.all(particles_t0[:, 3] == particles_t1[:, 3])
    data_t0 = particles_t0[:, 0:2]
    data_t1 = particles_t1[:, 0:2]
    
    shuffle = np.random.permutation(data_t0.shape[0])
    data_t0 = data_t0[shuffle, :]
    data_t1 = data_t1[shuffle, :]

    displacements     = np.full((len(groupsizes)), np.nan)
    num_displacements = np.full((len(groupsizes)), 0)

    for groupsize_index in numba.prange(len(groupsizes)):
        groupsize = groupsizes[groupsize_index]
        if particle_ids_at_these_timesteps_size < groupsize:
            continue

        num_groups = int(particle_ids_at_these_timesteps_size // groupsize)
        # assert num_groups > 0 # commented for numba
        displacements_at_time = np.full(num_groups, np.nan)
            
        for group_i in range(num_groups):
            print(group_i*groupsize, (group_i+1)*groupsize, group_i, groupsize)
            data_this_group_t0 = data_t0[group_i*groupsize:(group_i+1)*groupsize, :]
            data_this_group_t1 = data_t1[group_i*groupsize:(group_i+1)*groupsize, :]

            # assert data_this_group_t0.shape[0] == groupsize

            # COM_start = data_this_group_t0[:, :].mean(axis=0)
            # COM_end   = data_this_group_t1[:, :].mean(axis=0)
            COM_start = common.numba_mean_2d_axis0(data_this_group_t0)
            COM_end   = common.numba_mean_2d_axis0(data_this_group_t1)


            displacement = np.sum((COM_end - COM_start)**2)
            # assert not np.isnan(displacement) # commented for numba
            displacements_at_time[group_i] = displacement

        
        displacements[groupsize_index] = displacements_at_time.mean()
        num_displacements[groupsize_index] += 1
    
    return displacements, num_displacements

def calc_centre_of_mass_onepoint(data, groupsize, num_time_origins):
    # data should be of shape (num particles) x (num timesteps) x (2)
    num_timesteps = data.shape[1]

    cores = 48
    if cores > 16:
        warnings.warn(f'using {cores} cores')

    num_time_origins = min(num_time_origins, num_timesteps)
    time_origins = [int(i) for i in np.linspace(0, num_timesteps-2, num_time_origins)]
    
    displacements = np.full(num_time_origins, np.nan)

    def get_data_at_timesteps():
        for time_origin_index in range(num_time_origins):
            time_origin = time_origins[time_origin_index]
            particles_these_timesteps = (~np.isnan(data[:, time_origin, 0])) & (~np.isnan(data[:, time_origin+1, 0]))
            data_these_timesteps = data[particles_these_timesteps, :, :][:, time_origin:time_origin+2, :]
            yield data_these_timesteps

    with multiprocessing.Pool(cores) as pool:
        task = functools.partial(calc_centre_of_mass_onepoint_for_single_timeorigin, groupsize)

        # all_data = []

        # for time_origin in tqdm.trange(num_timesteps-1, desc='preparing data'):
        #     data_these_timesteps = get_data_at_timestep(data, time_origin)
        #     all_data.append(data_these_timesteps)
        #     print('size', common.arraysize(data_these_timesteps, mult=len(all_data)))

        results = list(tqdm.tqdm(pool.imap(task, get_data_at_timesteps()), total=num_time_origins, desc=f'N={str(groupsize).ljust(3)}'))

        for i in range(len(results)):
            displacements[i] = np.mean(results[i])
            # common.term_hist(displacements_at_time)
        # break


    # print('data_this_group.shape', data_this_group.shape)
    return displacements.mean(), displacements.std()/np.sqrt(displacements.size)

def calc_centre_of_mass_onepoint_for_single_timeorigin(groupsize, data_these_timesteps):
    np.random.shuffle(data_these_timesteps) # shuffles along first axes (num particles)
        # groups = np.array_split(data_these_timetsteps, groupsize, axis=0)

    num_groups = data_these_timesteps.shape[0]//groupsize
    displacements_at_time = np.full(num_groups, np.nan)
        
    for group_i in range(num_groups):
        data_this_group = data_these_timesteps[group_i*groupsize:(group_i+1)*groupsize, :, :]

        assert data_this_group.shape[0] == groupsize

        COM_start = data_this_group[:, 0, :].mean(axis=0)
        COM_end   = data_this_group[:, 1, :].mean(axis=0)
            # COM_start = common.numba_mean_2d_axis0(data_this_group[:, 0, :])
            # COM_end   = common.numba_mean_2d_axis0(data_this_group[:, 1, :])


        displacement = np.sum((COM_end - COM_start)**2)
        displacements_at_time[group_i] = displacement
    return displacements_at_time




# @numba.njit(parallel=True)
# def calc_centre_of_mass_onepoint_numba(data, groupsize):
#     print(numba.config.THREADING_LAYER)
#     # data should be of shape (num particles) x (num timesteps) x (2)
#     num_timesteps = data.shape[1]

#     displacements = np.full(num_timesteps-1, np.nan)

#     for time_origin in numba.prange(num_timesteps-1):
#         particles_these_timesteps = (~np.isnan(data[:, time_origin, 0])) & (~np.isnan(data[:, time_origin+1, 0]))
#         data_these_timesteps = data[particles_these_timesteps, :, :][:, time_origin:time_origin+2, :]

#         np.random.shuffle(data_these_timesteps) # shuffles along first axes (num particles)
#         # groups = np.array_split(data_these_timetsteps, groupsize, axis=0)

#         num_groups = data_these_timesteps.shape[0]//groupsize
#         displacements_at_time = np.full(num_groups, np.nan)
        
#         for group_i in range(num_groups):
#             data_this_group = data_these_timesteps[group_i*groupsize:(group_i+1)*groupsize, :, :]

#             # assert data_this_group.shape[0] == groupsize

#             # COM_start = data_this_group[:, 0, :].mean(axis=0)
#             # COM_end   = data_this_group[:, 1, :].mean(axis=0)
#             COM_start = common.numba_mean_2d_axis0(data_this_group[:, 0, :])
#             COM_end   = common.numba_mean_2d_axis0(data_this_group[:, 1, :])


#             displacement = np.sum((COM_end - COM_start)**2)
#             displacements_at_time[group_i] = displacement

#         displacements[time_origin] = np.mean(displacements_at_time)
#         # common.term_hist(displacements_at_time)
#         # break


#     # print('data_this_group.shape', data_this_group.shape)
#     return displacements.mean(), displacements.std()

def is_integer(arr):
    return arr % 1 == 0

def calc_incremental_xyz(particles, num_dimensions):
    # version that doesn't need reshaping - probably slower but a lot less memory
    # needs the time column to be 1-spaced integers (frame number)

    time_column = num_dimensions
    id_column = num_dimensions + 1

    assert particles[:, id_column].min() == 0

    assert np.all(is_integer(particles[:, time_column])), 'times should be all integer when using this method'
    assert np.all(np.diff(np.unique(particles[:, time_column])) == 1), 'times seems to be non-contignous'

    # need the data are sorted by particle ID
    particles = particles[particles[:, id_column].argsort()]
    
    num_particles = int(particles[:, id_column].max()) + 1
    num_timesteps = int(particles[:, time_column].max()) + 1
    print(f'{num_particles} particles, {num_timesteps} timesteps')

    msd_sum    = np.zeros((num_dimensions, num_timesteps))
    msd_sum_sq = np.zeros((num_dimensions, num_timesteps))
    msd_count  = np.zeros(num_timesteps)

    progress = tqdm.tqdm(total=particles.shape[0])

    start_index = 0
    current_id = 0
    skipped = 0

    # calc_every = 10

    while start_index < particles.shape[0]-1:
        current_id = particles[start_index, id_column]
        end_index = start_index + 1
        # print(start_index, end_index, particles.shape[0])
        # find the index of the last row that has this particle ID
        while end_index < particles.shape[0] and particles[end_index, id_column] == current_id:
            end_index += 1

        data_this_particle = particles[start_index:end_index, :]
        assert np.all(data_this_particle[:, id_column] == current_id)
        
        # now sort by time
        data_this_particle = data_this_particle[data_this_particle[:, time_column].argsort()]

        assert not np.any(np.isnan(data_this_particle))

        if data_this_particle.shape[0] == 0:
            raise Exception('what is this doing?!!!!')
            pass

        # elif current_id % calc_every != 0:
        #     pass

        else:
            # print(data_this_particle[:, 2])
            # assert data_this_particle[0, 2] == 0
            # num_timesteps_this_particle = int(data_this_particle[:, 2].max()) + 1
            num_timesteps_this_particle = data_this_particle.shape[0]
            # print(num_timesteps_this_particle, data_this_particle.shape)
            # print(data_this_particle)
            # [print(data_this_particle[i, :]) for i in range(data_this_particle.shape[0])]
            # print(num_timesteps_this_particle, data_this_particle.shape[0])

            if np.any(np.diff(data_this_particle[:, time_column]) != 1):
                # this means that the timestep was non-contiguous
                skipped += 1
                # print(num_timesteps_this_particle, data_this_particle.shape[0])
            
            else:
                # print(num_timesteps_this_particle)
                # assert data_this_particle[-1, 2] == num_timesteps_this_particle - 1
                # t1 = time.time()
                MSDs = np.full((num_dimensions, data_this_particle.shape[0]), np.nan)
                for dimension in range(num_dimensions):
                    MSDs[dimension, :] = msd_fft1d(data_this_particle[:, dimension])
                # MSD = MSDs.sum(axis=0)
                assert not np.any(np.isnan(MSDs))
                # t2 = time.time()
                msd_sum   [:, :num_timesteps_this_particle] += MSDs
                msd_sum_sq[:, :num_timesteps_this_particle] += MSDs**2
                msd_count[:num_timesteps_this_particle] += 1
                # t3 = time.time()
                # t_a = t1-t0
                # t_b = t2-t1
                # t_c = t3-t2
                # t = t3-t0
                # print(f'{t_a/t:.2f} {t_b/t:.2f} {t_c/t:.2f}')
        progress.n = end_index # https://github.com/tqdm/tqdm/issues/1264
        progress.refresh()
        start_index = end_index
    progress.close()

    print(f'skipped {skipped/num_particles:.2f}')
    assert skipped/num_particles < 0.5, f'{skipped/num_particles} of particles were skipped'
    # assert skipped/num_particles < 0.7


    assert not np.any(np.isnan(msd_sum))
    assert not np.any(np.isnan(msd_count))

    assert not np.all(msd_count == 0)
    
    if np.any(msd_count == 0):
        raise NotImplementedError('this has not been updated since we moved to calculating each dimension')
        index = np.argmax(msd_count==0)
        print(f'no particles were found after frame={index}')
        if index < 100:
            raise Exception(f'no particles were found after frame={index}')
        msd_sum    = msd_sum   [:index]
        msd_sum_sq = msd_sum_sq[:index]
        msd_count  = msd_count [:index]
    
    avg = msd_sum / msd_count
    std = msd_sum_sq / msd_count - avg**2

    assert np.isclose(avg[:, 0], 0, atol=1e-3).all(), f'msd = {avg[:, 0]} at frame 0'
    # this atol should maybe depend on if it's translational or rotational diffusion I think
    # because rotational diffusions in rad are of smaller magnitude than translationals in um (is this definitely true?)
    # it seems to also depend on the magnitude of the timestep

    assert not np.any(np.isnan(avg))

    unc = std / np.sqrt(msd_count)
    # unc = std

    return avg, unc

def calc_incremental(particles, num_dimensions):
    msds, msds_unc = calc_incremental_xyz(particles, num_dimensions)
    return msds.sum(axis=0), msds_unc.sum(axis=0)

"""
@numba.njit()
def calc_msd_for_shift(data, shift_size, shift_length, exponent, shift_index):
    # data should be of shape (num particles) x (num timesteps) x (2)
    assert data.shape[0] > 1 # num_particles == 1 fails because the dimension gets squeezed
    # t0 = time.time()
    
    shift = shift_index * shift_size
    assert shift+shift_length <= data.shape[1], 'shift_index * shift_size + shift_length was longer than num_timesteps'
    
    data_to_consider = data[:, shift:shift+shift_length, :]
    # below loop is numba compatible replacement of this line:
    # is_nan = np.any(np.isnan(data_to_consider), axis=(1, 2))
    is_nan = np.full((data_to_consider.shape[0]), True)
    for particle_i in range(0, data_to_consider.shape[0]):
        is_nan[particle_i] = np.any(np.isnan(data_to_consider[particle_i, :, 0])) # only test x-value

    #print(f"dropping {is_not_nan.sum()/data_to_consider.shape[0]*100:.0f}%")
                        
    data_to_use = data_to_consider[~is_nan, :, :]

    r0 = data_to_use[:, 0, :]
    assert r0.size != 0
    assert np.isnan(r0).sum() == 0
    r0 = r0[:, np.newaxis, :] # could remove this line if r0 = data_to_use[:, [0], :]

    # with viztracer.get_tracer().log_event("test1"):
    displacements = (data_to_use - r0) # r0 shape is broadcasted up to data_to_use shape

    # with viztracer.get_tracer().log_event("test2"):
    square_displacements = displacements[:, :, 0]**2 + displacements[:, :, 1]**2

    # square_displacements = square_displacements ** (exponent/2)

    # below loop is numba-compatibe equivalent of this line:
    # msd = square_displacements.mean(axis=0)
    msd = np.full((shift_length), np.nan)
    for time_i in range(0, shift_length):
        msd[time_i] = 1
        msd[time_i] = square_displacements[:, time_i].mean()

    #mean_displacement = displacements.mean(axis=0)
    #print(mean_displacement.shape)
    #mean_displacement_length = np.sqrt(mean_displacement[:, 0]**2 + mean_displacement[:, 1]**2)

    # msds[shift_index, :] = msd
    #mds [shift_index, :] = mean_displacement_length
    # print('done')
    # progress.tick()
    # t1 = time.time()
    # print(f'{(t1 - t0)*100}ms')
    return msd

# @viztracer.log_sparse(stack_depth=3)
# @viztracer.log_sparse()
@numba.njit(parallel = True)
# @numba.njit()
def MSD(data, num_shifts, shift_length, time_step, exponent=2):
    # data should be of shape (num particles) x (num timesteps) x (2)
    num_timesteps = data.shape[1]

    assert num_shifts > 0
    assert shift_length > 0

    shift_size = int((num_timesteps - shift_length) / num_shifts)
    
    assert shift_size > 0
    shift_overlap = (shift_length-shift_size)/shift_length
    shift_overlap = shift_overlap if shift_overlap > 0 else 0
    # print(f"{num_shifts} shifts of size {shift_size} (length = {shift_length}, overlap = {shift_overlap*100:.2f}%)")
    print(num_shifts, 'shifts of size', shift_size, '(length =', shift_length, 'overlap =', shift_overlap, ')')

    msds = np.full((num_shifts, shift_length), np.nan)# * units.micrometer**exponent

    # we could get extra data at low t if we wanted, at the moment we average over the same number
    # regardless of how big or small t is
    
    parallel = False

    print('starting main MSD section')

    if not parallel:
        # progress = Progress(num_shifts)
        # times = []
        for shift_index in numba.prange(0, num_shifts):
            c = calc_msd_for_shift(data, shift_size, shift_length, exponent, shift_index)
            msds[shift_index, :] = c# * units.micrometer**2
            # times.append(c[1])
            # progress.tick()
            print(shift_index, '/', num_shifts)
        # print(f'time: {np.mean(times)*1000:.0f} +- {np.std(times)*1000:.0f} ms')
        # progress.done()

    # else:
    #     chunksize = 9
    #     print(f'{multiprocessing.cpu_count()} cores, {math.ceil(num_shifts/chunksize)} chunks of size {chunksize}')
    #     print(f'data size: {arraysize(data)}')
    #     with multiprocessing.Pool() as pool:

    #         # def task_done(result):
    #         #     progress.tick()
    #         #     print('tick', flush=True)

    #         # shared_array_base = multiprocessing.Array(ctypes.c_double, int(np.prod(data.shape)))
    #         # data_shared = np.ctypeslib.as_array(shared_array_base.get_obj())
    #         # data_shared = data_shared.reshape(*data.shape)
    #         # data_shared[:, :, :] = data

    #         task = functools.partial(calc_msd_for_shift, data, shift_size, shift_length, exponent)

    #         list_of_msds = tqdm.tqdm(pool.imap(task, range(0, num_shifts), chunksize=chunksize), total=num_shifts)
    #         #print(type(list_of_msds))

    #         list_of_msds = list(list_of_msds)
    #         # print(type(a))
    #         # print(type(a[0]), a[0])
    #         # b = np.array(list_of_msds)
    #         # print(type(b), b)
    #         # print(type(list_of_msds))
    #         # print(type(list_of_msds[0]))

    #         # while not task.ready()

    #         # list_of_msds = tasks.get()
    #         times = []

    #         # now turn that list of np arrays back into a 2d np array
    #         for i in range(0, num_shifts):
    #             msds[i, :] = list_of_msds[i][0] * units.micrometer**2
    #             times.append(list_of_msds[i][1])

    #         print(f'time: {np.mean(times)*1000:.0f} +- {np.std(times)*1000:.0f} ms')
            


    #mean_msd = msds.mean(axis=0)
    #mean_md  = mds .mean(axis=0)
            
    print('finished main MSD section')

    mean_msd = np.full((shift_length), np.nan)#~np.any(np.isnan(data_to_consider), axis=(1, 2))  # !viztracer: log
    for time_i in numba.prange(0, shift_length):
        mean_msd[time_i] = msds[:, time_i].mean()

    print('finished means')

    t = np.arange(0, mean_msd.size) * time_step

    t        = t       [~np.isnan(mean_msd)]
    mean_msd = mean_msd[~np.isnan(mean_msd)]

    return t, mean_msd
    """

###### msd mixt seems complicated so we're gonna start again
# knowing that we only need to calculate MSD(frame=1)


def get_frames_with_delta_t(t, dt):
    # find all pairs of frames in `t` separated by `dt`
    pairs = []

    print('t[:10]', t[:10])

    pairs = np.zeros((len(t), 2), dtype=int)
    num_pairs_found = 0

    for frame in range(len(t)):
        if len(index := np.where(t == t[frame] + dt)[0]):

            pairs[num_pairs_found, :] = (frame, index[0])
            num_pairs_found += 1

    pairs = pairs[:num_pairs_found, :]

    assert len(pairs), f'no frames with time delta {dt} were found in the data'

    assert np.isfinite(pairs).all()
    return pairs

def get_particles_at_frame(particles):
    ##### this was shamefully copied straight from scattering functions

    # assert particles.dtype == np.float32

    # data is x,y,t
    particles[:, 2] -= particles[:, 2].min() # convert time to being 0-based

    # find max number of particles at any one timestep
    times, num_particles_at_frame = np.unique(particles[:, 2], return_counts=True)
    num_timesteps = times.size
    max_particles_at_frame = num_particles_at_frame.max()
    assert num_particles_at_frame.size == num_timesteps, f'{num_particles_at_frame.size} != {num_timesteps}'


    ###########start

    #### this whole thing a mess and tbh I don't remember why I did it this way
    #### ffs adam this is a sign that you need to take more time over the code

    # for Fself, we need the IDs, so we provide a list where the nth element is the nth particle
    assert particles.shape[1] == 4, 'for self intermediate scattering, you should provide rows of x,y,t,#'

    # we sort the ting by particle ID, then time
    print('sorting')
    particles = particles[particles[:, 3].argsort()]
    particles = particles[np.lexsort((particles[:, 2], particles[:, 3]))] # sort by ID then time
    print('sorted')

    # for the reshaping we use frame number not time
    time_to_frame = lambda time: np.argmax(times == time)
    assert time_to_frame(0) == 0
    for row in tqdm.trange(particles.shape[0], desc='mapping times'):
        particles[row, 2] = time_to_frame(particles[row, 2])

    assert particles.shape[0], 'particles was empty'

    # find trajectories that are long enough
    i = 0
    MIN_TRAJ_LENGTH = 0
    good_ids = []
    progress = tqdm.tqdm(total=particles.shape[0], desc='finding trajectories')
    skipped = 0
    too_short = 0

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

        if rows_used != traj_length + 1: # this essentially checks that the frame number is going up by one each time
            skipped += 1
                
            if start_time == 0:
                print('start t zero skipped')
        else:
            if traj_length > MIN_TRAJ_LENGTH:
                good_ids.append((current_id, start_i, end_i))
                if start_time == 0:
                    print('start t zero good')
            else:
                too_short += 1
                if start_time == 0:
                    print('start t zero too short')
        
        i += 1

    progress.close()

    assert len(good_ids), 'good_ids is empty'

    print('good trajs', len(good_ids)/particles[:, 3].max())
    print('skipped', skipped/particles[:, 3].max())
    print('too short', too_short/particles[:, 3].max())

    print(particles[:10, :])

    # save these trajectories in a new shape - (num timesteps) x (num trajectories) x 2
    particles_at_frame = np.full((num_timesteps, len(good_ids), 2), np.nan)
    xys = particles[:, [0, 1]]
    for id_index in tqdm.trange(len(good_ids), desc='reshaping'):
        old_id = good_ids[id_index][0]
        new_id = id_index # need to resample the IDs when we remove some particles


        start_i = good_ids[id_index][1]
        end_i   = good_ids[id_index][2]
        start_t = int(particles[start_i, 2])
        end_t   = int(particles[end_i-1, 2])

        if id_index < 5:
            print(old_id, good_ids[id_index], start_i, end_i, start_t, end_t)
        
        xys_to_save = xys[start_i:end_i, :]
        assert xys_to_save.shape[0], 'there are no rows in xys_to_save'
        assert np.isfinite(xys_to_save).all()
        # print(xys_to_save.shape)
        # print(particles_at_frame[start_t:end_t+1, new_id, :].shape)
        particles_at_frame[start_t:end_t+1, new_id, :] = xys_to_save

        # if i > 10:
        #     break
    
    # print(np.isfinite(particles_at_frame).all(axis=(1, 2)).sum())
    # print(np.isfinite(particles_at_frame).all(axis=(0, 2)).sum())

    num_particles = int(particles[:, 3].max()) + 1
    assert num_particles > 0

    del particles

    # check that for every particle there is at least one non-nan coordinate
    assert np.isfinite(particles_at_frame).any(axis=(0, 2)).all()
    # check that for every time there is at least one non-nan coordinate
    assert np.isfinite(particles_at_frame).any(axis=(1, 2)).all()
    print(np.isfinite(particles_at_frame).any(axis=2).sum(axis=1))
    print('min particles at frame', np.min(np.isfinite(particles_at_frame).any(axis=2).sum(axis=1)))
    assert np.all(np.isfinite(particles_at_frame).any(axis=2).sum(axis=1) > 10)

    # raise Exception('you have not yet implemented the new time (not frame) based approach here')


    warnings.warn('this needs to be back-copied to scattering_functions')

    ################end

    assert particles_at_frame.shape[0] == times.shape[0]
    assert np.isfinite(times).all()

    return particles_at_frame, times

def calc_mixt(particles, requested_times, max_time_origins):

    msd     = np.full(len(requested_times), np.nan)
    msd_unc = np.full(len(requested_times), np.nan)

    particles_at_frame, times = get_particles_at_frame(particles)

    for time_i, requested_time in enumerate(tqdm.tqdm(requested_times, desc='calculating')):
        pairs = get_frames_with_delta_t(times, requested_time)
        assert len(pairs)

        use_every_nth_pair = int(np.ceil(pairs.shape[0] / max_time_origins))
        print(f'd_frame = {requested_time} : use_every_nth_pair = {use_every_nth_pair}')

        pairs_to_use = pairs[::use_every_nth_pair, :]

        r2     = np.full(pairs_to_use.shape[0], np.nan)
        r2_std = np.full(pairs_to_use.shape[0], np.nan)

        for pair_i in range(pairs_to_use.shape[0]):
            frame1, frame2 = pairs_to_use[pair_i, :]
            # if frame1 > 1:
            #     continue
            # print(frame1, frame2)
            r2[pair_i], r2_std[pair_i] = calc_for_pair(particles_at_frame[frame1, :, :], particles_at_frame[frame2, :, :])

        assert np.isfinite(r2).all()

        msd    [time_i] = r2.mean()
        msd_unc[time_i] = r2_std.mean()

        if requested_time > 0:
            assert msd[time_i] > 0

    assert np.isfinite(msd).all()

    return msd, msd_unc
        
def calc_for_pair(xy_t0, xy_t1):
    # for two lists of coordinates, calculate the mean and std of the change in positions

    assert xy_t0.shape[0], 'xy_t0 has no rows in it'
    assert xy_t1.shape[0], 'xy_t1 has no rows in it'


    # there can be nans if a particle has moved off the screen between frames, so we remove any such particles
    t0_mask = np.isfinite(xy_t0).all(axis=1)
    t1_mask = np.isfinite(xy_t1).all(axis=1)
    assert t0_mask.sum(), 'no particles were found at t0'
    assert t1_mask.sum(), 'no particles were found at t1'
    mask = t0_mask & t1_mask
    assert mask.sum(), f'no particles were found in the intersection of t0 & t1. t0 has {np.isfinite(xy_t0).all(axis=1).sum()} and t1 {np.isfinite(xy_t1).all(axis=1).sum()}'
    xy_t0_safe = xy_t0[mask, :]
    xy_t1_safe = xy_t1[mask, :]

    r_xy = xy_t1_safe - xy_t0_safe
    r2 = r_xy[:, 0]**2 + r_xy[:, 1]**2
    assert np.isfinite(r2).all()
    
    mean, std = r2.mean(), r2.std()
    assert np.isfinite(mean)
    return mean, std

    

def calc_mixt_new(particles, requested_times, dimension):
    # new philosophy: no reshaping

    time_column = dimension
    id_column = dimension + 1

    # we sort the ting by particle ID, then time
    print('sorting')
    particles = particles[particles[:, id_column].argsort()]
    particles = particles[np.lexsort((particles[:, time_column], particles[:, id_column]))] # sort by ID then time
    print('sorted')

    # we use the iterative mean/std dev method
    num        = np.zeros(len(requested_times))
    msd_sum    = np.zeros(len(requested_times))
    msd_sq_sum = np.zeros(len(requested_times))

    row = 0

    progress = tqdm.tqdm(total=particles.shape[0])

    while row < particles.shape[0]:
        particle_id = particles[row, id_column]

        while row < particles.shape[0] and particles[row, id_column] == particle_id:
            time_origin = particles[row, time_column]
            origin = particles[row, 0:dimension]

            row_time_destination = row # this is the row for the time we're comparing to

            for time_index in range(len(requested_times)):
                requested_time = requested_times[time_index]

                while row_time_destination < particles.shape[0] and particles[row_time_destination, id_column] == particle_id and particles[row_time_destination, time_column] - time_origin <= requested_time:
                    # print('checking time destination', particles[row_time_destination, 2])
                    if particles[row_time_destination, time_column] - time_origin == requested_time:
                        # we found the right time delta, so we can use this
                        destination = particles[row_time_destination, 0:dimension]
                        r2 = ((destination - origin)**2).sum()


                        num       [time_index] += 1
                        msd_sum   [time_index] += r2
                        msd_sq_sum[time_index] += r2**2

                    row_time_destination += 1
                
            row += 1
            progress.update()

            # I think there is a problem here in that if

    progress.close()

    msd = msd_sum / num
    msd_std = np.sqrt(msd_sq_sum/num - (msd_sum/num)**2)

    if np.any(num == 0):
        raise Exception(f'no points were found with time delta {np.array(requested_times)[num == 0]}')

    return msd, msd_std

    

def calc_mixt_new_xyz(particles, requested_times, num_dimensions):
    # new philosophy: no reshaping
    print('THIS METHOD DOES VERY BADLY COMPARED TO calc_incremental')

    ID_COLUMN = num_dimensions + 1
    TIME_COLUMN = num_dimensions

    # we sort the ting by particle ID, then time
    print('sorting')
    particles = particles[particles[:, ID_COLUMN].argsort()]
    particles = particles[np.lexsort((particles[:, TIME_COLUMN], particles[:, ID_COLUMN]))] # sort by ID then time
    print('sorted')

    # we use the iterative mean/std dev method
    num        = np.zeros(len(requested_times))
    msd_sum    = np.zeros((num_dimensions, len(requested_times)))
    msd_sq_sum = np.zeros((num_dimensions, len(requested_times)))

    row = 0

    progress = tqdm.tqdm(total=particles.shape[0])

    while row < particles.shape[0]:
        particle_id = particles[row, ID_COLUMN]

        while row < particles.shape[0] and particles[row, ID_COLUMN] == particle_id:
            time_origin = particles[row, TIME_COLUMN]
            origin_coord = particles[row, 0:num_dimensions]
            # print('doing time origin', time_origin)

            row_time_destination = row # this is the row for the time we're comparing to

            for time_index in range(len(requested_times)):
                requested_time = requested_times[time_index]

                while row_time_destination < particles.shape[0] and particles[row_time_destination, ID_COLUMN] == particle_id and particles[row_time_destination, TIME_COLUMN] - time_origin <= requested_time:
                    # print('checking time destination', particles[row_time_destination, 2])
                    if particles[row_time_destination, TIME_COLUMN] - time_origin == requested_time:
                        # we found the right time delta, so we can use this
                        destination_coord = particles[row_time_destination, 0:num_dimensions]

                        num[time_index] += 1

                        for dimension in range(num_dimensions):
                            r2 = (origin_coord[dimension] - destination_coord[dimension])**2
                            msd_sum   [dimension, time_index] += r2
                            msd_sq_sum[dimension, time_index] += r2**2

                    row_time_destination += 1
                
            row += 1 # you could change this to += 2 for use_every_nth_timestep = 2 etc
            progress.update()

            # I think there is a problem here in that if

    progress.close()

    msd = msd_sum / num
    msd_std = np.sqrt(msd_sq_sum/num - (msd_sum/num)**2)

    if np.any(num == 0):
        raise Exception(f'no points were found with time delta {np.array(requested_times)[num == 0]}')

    return msd, msd_std
