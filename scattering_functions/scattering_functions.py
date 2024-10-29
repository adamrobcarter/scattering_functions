import numpy as np
import scipy
import scipy.stats
import time
import functools
import tqdm
import os.path, time, datetime
import multiprocessing
import scipy.fft
import common
import warnings
import numba
"""
# def theoretical_c_hat(k, rho, sigma):
#     # sigma is disk diameter
#     phi = np.pi/4 * rho * sigma**2
#     J0 = lambda x: scipy.special.jv(0, x.magnitude)
#     J1 = lambda x: scipy.special.jv(1, x.magnitude)

#     prefactor = np.pi / ( 6 * ( 1 - phi)**3 * k**2 )
#     line1 = -5/4 * (1 - phi)**2 * k**2 * sigma**2 * J0(k * sigma / 2)**2
#     line23 = 4 * ( (phi - 20) * phi + 7) + 5/4 * (1 - phi)**2 * k**2 * sigma**2
#     line23factor = J1(k * sigma / 2)**2
#     line4 = 2 * (phi - 13) * (1 - phi) * k * sigma * J1(k * sigma / 2) * J0(k * sigma / 2)
#     c = prefactor * (line1 + line23*line23factor + line4)
#     return c

# def S_from_c_hat(c_hat, rho):
#     # Hansen & McDonald (3.6.10)
#     # rho is bulk fluid number density
#     S = 1 / (1 - rho * c_hat)
#     return S

# def h_hat_from_c_hat(c_hat, rho):
#     # Hansen & McDonald (3.5.13)
#     return c_hat / (1 - rho * c_hat)

# def h_hat_from_g_hat(g_hat):
#     # Hansen & McDonald (2.6.8)
#     h_hat = g_hat
#     #h_hat[0] = 1
#     return h_hat

# def h_from_g(g):
#     # Hansen & McDonald (2.6.8)
#     return g - 1

# def g_from_h(g):
#     # Hansen & McDonald (2.6.8)
#     return g + 1

# def S_from_h_hat(h_hat, rho):
#     # Hansen & McDonald (3.6.10)
#     return 1 + rho * h_hat

# def density_correlation(locations, max_r, num_r_bins, width, height, crop_border):
#     r_bin_edges = unit_linspace(0*units.meter, max_r, num_r_bins)
#     r_bin_width = r_bin_edges[1] - r_bin_edges[0]
#     densities = np.zeros((num_r_bins,)) * units.micrometer**(-2)

#     x = locations[:, 0]
#     y = locations[:, 1]
#     out_of_crop = (x < crop_border) | (x > (width -crop_border)) | (y < crop_border) | (y > (height-crop_border))
#     cropped_fraction = out_of_crop.sum() / x.size
#     assert cropped_fraction < 1, 'all particles were cropped out'
#     # used_locations = np.delete(locations, out_of_crop, axis=0)
#     used_locations = locations[~out_of_crop, :]
#     num_used_particles = used_locations.shape[0]

#     # precalculate distance in one go
#     used_locations_x = used_locations[:, 0]
#     used_locations_y = used_locations[:, 1]
#     locations_x = locations[:, 0]
#     locations_y = locations[:, 1]
#     dx = used_locations_x[:, np.newaxis] - locations_x[np.newaxis, :]
#     dy = used_locations_y[:, np.newaxis] - locations_y[np.newaxis, :]
#     r = np.sqrt(dx**2 + dy**2) #  r is num_particles x num_all_particles  where elements are the distance

#     for bin_index, left_edge in enumerate(r_bin_edges):
#         right_edge = left_edge + r_bin_width
#         # need to find number of particles in donut of inner radius left_edge
#         # and outer radius right_edge
        
#         num_donuts = num_used_particles
#         assert num_donuts > 0
#         donut_area = np.pi * (right_edge**2 - left_edge**2) # approx = (left_edge + r_bin_width/2) * r_bin_width

#         num_particles_in_donuts = np.count_nonzero(np.logical_and(left_edge <= r, r < right_edge))

#         avg_particles_per_donut = num_particles_in_donuts / num_donuts
#         densities[bin_index] = avg_particles_per_donut / donut_area

#     avg_density = num_used_particles / ( (width - 2*crop_border) * (height - 2*crop_border) )
#     densities = densities / avg_density
#     return r_bin_edges, densities, avg_density

def intermediate_scattering(log, F_type, crop, num_k_bins, num_iters, d_frames, data, num_frames, max_K, width, height):
    assert not np.isnan(max_K)

    # data is x,y,t
    data[:, 2] -= data[:, 2].min() # convert time to being 0-based

    Fs    = np.full((len(d_frames), num_k_bins), np.nan)
    F_unc = np.full((len(d_frames), num_k_bins), np.nan)
    print(f'F size {common.arraysize(Fs)}')
    ks   = np.full((len(d_frames), num_k_bins+1), np.nan) # +1 because we get the bin edges

    min_K = 2*np.pi/( min(width, height) * crop )

    # first find the particles at each timestep, otherwise we're transferring
    # the whole of data to each process
    particles_at_frame = []
    for frame in range(num_frames):
        # select only particles at the relevent time step
        particles = data[data[:, 2]==frame, :]
        # select only x and y columns
        particles = particles[:, 0:2]
        particles_at_frame.append(particles)

    parallel = False
    if parallel:
        pass
        # below is old multiprocessing implementation
        # with multiprocessing.Pool(32) as pool:

        #     internal_bound = functools.partial(intermediate_scattering_for_dframe, log=log, F_type=F_type, crop=crop,
        #                         num_k_bins=num_k_bins, num_iters=num_iters, d_frames=d_frames, data=data,
        #                         num_frames=num_frames, min_K=min_K, max_K=max_K, width=width, height=height)

        #     computation = pool.imap(internal_bound, range(len(d_frames)), chunksize=1)
        #     results = list(tqdm.tqdm(computation, total=len(d_frames)))
            
        #     for dframe_i in range(len(d_frames)):
        #         F_, F_unc_, k_ = results[dframe_i]
        #         Fs   [dframe_i, :] = F_
        #         F_unc[dframe_i, :] = F_unc_
        #         ks   [dframe_i, :] = k_

    else:
        for dframe_i in tqdm.trange(len(d_frames)):
            F_, F_unc_, k_ = intermediate_scattering_for_dframe(dframe_i, log=log, F_type=F_type, crop=crop,
                                num_k_bins=num_k_bins, num_iters=num_iters, d_frames=d_frames, particles_at_frame=particles_at_frame,
                                num_frames=num_frames, min_K=min_K, max_K=max_K, width=width, height=height)
            
            nan_fraction = np.isnan(F_).sum() / F_.size
            assert nan_fraction < 0.5, f'F nan fraction was {nan_fraction*100:.0f}%'
            
            nan_fraction = np.isnan(k_).sum() / k_.size
            assert nan_fraction < 0.5, f'k nan fraction was {nan_fraction*100:.0f}%'
            
            Fs   [dframe_i, :] = F_
            F_unc[dframe_i, :] = F_unc_
            ks   [dframe_i, :] = k_

    nan_fraction = np.isnan(Fs).sum() / Fs.size
    assert nan_fraction < 0.5, f'F nan fraction was {nan_fraction*100:.0f}%'

    nan_fraction = np.isnan(ks).sum() / ks.size
    assert nan_fraction < 0.5, f'k nan fraction was {nan_fraction*100:.0f}%'

    # now remove irrelevent data
    # min_useful_K = 2*np.pi/min(width, height)
    # print('exit    ', ks[0, :6])
    # print('exit    ', ks[0, :6] <= min_K)
    # print(min_useful_K, min_K)
    Fs[ks[:, :-1] <= min_K] = np.nan 
    Fs[ks[:,  1:] >  max_K] = np.nan # remove any greater than max_K (limited data here as get only diagonals)
    #  ks[:, 1:] - this is because we wanna test with the left/right bin edge

    # the first k bin is always all nan, so remove it
    assert np.isnan(Fs[:, 0]).sum() == Fs[:, 0].size
    ks    = ks   [:, 1:]
    Fs    = Fs   [:, 1:]
    F_unc = F_unc[:, 1:]

    # print('min_K', )
    
    ks = (ks[:, 1:] + ks[:, :-1])/2 # return bin midpoints not edges

    return Fs,F_unc,ks

@numba.njit(parallel=True)
# @numba.njit()
def intermediate_scattering_for_dframe(dframe_i, log, F_type, crop, num_k_bins, num_iters, d_frames, particles_at_frame, num_frames, min_K, max_K, width, height):
    d_frame = int(d_frames[dframe_i])
    # offset = (num_frames - d_frame - 1) // num_iters
    # assert(num_iters * offset + d_frame < num_frames)

    frames_to_use = list(range(0, num_frames-d_frame, 10))
    num_used_frames = len(frames_to_use)
    print('num used frames', num_used_frames)

    F = np.full((num_used_frames, num_k_bins),    np.nan)
    k = np.full((num_used_frames, num_k_bins+1,), np.nan) # +1 b/c we get the left and right of the final bin
                    
    if F_type == 'F_s' or F_type == 'Fs':
        # func = self_intermediate_scattering_internal
        raise Exception("not yet implemented")
    else:
        # func = intermediate_scattering_internal
        pass

    for frame_index in numba.prange(num_used_frames):
    # for frame in range(num_frames):
        frame = frames_to_use[frame_index]

        # particles_t0 = data[:, frame, :]
        # particles_t1 = data[:, frame + d_frame, :]
        particles_t0 = particles_at_frame[frame]
        particles_t1 = particles_at_frame[frame + d_frame]
        
        particles_t0, particles_t1 = preprocess_scattering(particles_t0, particles_t1, crop=crop)
        width  = width  * crop
        height = height * crop
    
        k_x, k_y, k_bins = get_k_and_bins_for_intermediate_scattering(min_K, max_K, num_k_bins, log_calc=log, log_bins=log)

        k_unbinned, F_unbinned = intermediate_scattering_internal(particles_t0, particles_t1, k_x, k_y)
        
        k_, F_ = postprocess_scattering(k_unbinned, F_unbinned, k_bins)
        
        F[frame_index, :] = F_
        k[frame_index, :] = k_

    # check all returned ks are the same
    # for frame in range(1, num_used_frames):
    #     assert k[frame, :] == k[0, :]

    assert np.isnan(F).sum() < F.size

                # print(f"nan S: {np.isnan(F).sum()/F.size:.2f}, nan k: {np.isnan(k).sum()/k.size:.2f}")
                #assert(np.isnan(S).sum()/S.size < 0.5)
                #print(f'min_K={min_K:.3f}, k bin size={k_[1]-k_[0]:.3f}, num bins={num_k_bins}')

    # need nanmean because binned_statistic will return nan if the bin is empty
    # return np.nanmean(F, axis=0), np.nanstd(F, axis=0), k
    return common.numba_nanmean_2d_axis0(F), common.numba_nanstd_2d_axis0(F), k[0, :]

@numba.njit
def preprocess_scattering(particles_t0, particles_t1, crop):
    # particles_t(0/1) are of shape (num_particles x 2)

    # first remove any nan particles (particles that exist at t0 but not t1)
    # t0_nans = np.any(np.isnan(particles_t0), axis=1)
    # t1_nans = np.any(np.isnan(particles_t1), axis=1)
    # nans = t0_nans | t1_nans
    # # print(f'missing particles: {nans.sum()/nans.size*100}%')
    # particles_t0 = particles_t0[~nans, :]
    # particles_t1 = particles_t1[~nans, :]

    assert np.isnan(particles_t0).sum() == 0
    assert np.isnan(particles_t1).sum() == 0

    # then do the crop
    # width_thresh  = ( particles_t0[:, 0].max() - particles_t0[:, 0].min() ) * crop
    # height_thresh = ( particles_t0[:, 1].max() - particles_t0[:, 1].min() ) * crop
    # removed_particles_t0 = (particles_t0[:, 0] > width_thresh) | (particles_t0[:, 1] > height_thresh)
    # removed_particles_t1 = (particles_t1[:, 0] > width_thresh) | (particles_t1[:, 1] > height_thresh)
    # removed_particles = removed_particles_t0 | removed_particles_t1
    # particles_t0 = particles_t0[~removed_particles, :]
    # particles_t1 = particles_t1[~removed_particles, :]

    return particles_t0, particles_t1

@numba.njit
def postprocess_scattering(k, F, k_bins):
    common.numba_p_assert(np.isnan(F).sum() == 0, 'F was '+str(int(np.isnan(F).sum()/F.size*100))+'% NaN')

    # F_binned, k_binned, _ = scipy.stats.binned_statistic(k.flatten(), F.flatten(), 'mean', bins=k_bins)
    F_binned, k_binned, _ = common.numba_binned_statistic(k.flatten(), F.flatten(), bins=k_bins)
    # binned statistic returns NaN if the bin is empty
    
    common.numba_p_assert(np.isnan(F_binned).sum() < F_binned.size, 'F_binned was all nan '+str(np.isnan(F_binned).sum()) + ' ' + str(F_binned.size))
    common.numba_p_assert(np.isnan(k_binned).sum() < k_binned.size, 'k_binned was all nan')

    return k_binned, F_binned

@numba.njit
def get_k_and_bins_for_intermediate_scattering(min_K, max_K, num_k_bins, log_calc, log_bins):

    if log_calc:
        # k_x_pos =  np.logspace(np.log10(min_K), np.log10(max_K), num_k_bins, dtype='float64')
        # k_x_neg = -np.logspace(np.log10(min_K), np.log10(max_K), num_k_bins, dtype='float64')
        k_x_pos =  np.logspace(np.log10(min_K), np.log10(max_K), num_k_bins)
        k_x_neg = -np.logspace(np.log10(min_K), np.log10(max_K), num_k_bins)
        k_x = np.concatenate((k_x_neg, np.array([0]), k_x_pos))
        # k_y = np.logspace(np.log10(min_K), np.log10(max_K), num_k_bins, dtype='float64')
        k_y = np.logspace(np.log10(min_K), np.log10(max_K), num_k_bins)
        bin_edges = np.concatenate((np.array([0]), k_x_pos))
        # here we invent a bin 0 < k < min_K. Anything in here should be thrown away later
    else:
        # have checked this starting from min_K not -max_K and it does indeed seem to make no difference
        # k_x = np.arange(-max_K, max_K, min_K, dtype='float64')
        # k_y = np.arange( 0,     max_K, min_K, dtype='float64')
        k_x = np.arange(-max_K, max_K, min_K)
        k_y = np.arange( 0,     max_K, min_K)
        bin_edges = np.linspace(0, max_K, num_k_bins+1)

    common.numba_p_assert(np.isnan(k_x).sum() == 0, 'k_x had nan elements')
    common.numba_p_assert(np.isnan(k_y).sum() == 0, 'k_y had nan elements')

    return k_x, k_y, bin_edges

@numba.njit
def intermediate_scattering_internal(particles_t0, particles_t1, k_x, k_y):
    # Thorneywork et al 2018 eq (27)
    # particles_t(0/1) are of shape (num_particles x 2)

    num_particles_0 = particles_t0.shape[0]
    num_particles_1 = particles_t1.shape[0]
    common.numba_p_assert(num_particles_0 > 0, 'no particles were found in 0')
    common.numba_p_assert(num_particles_1 > 0, 'no particles were found in 1')
    
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

    k_x = k_x[np.newaxis, np.newaxis, :, np.newaxis]
    k_y = k_y[np.newaxis, np.newaxis, np.newaxis, :]
    
    # TODO: do we not need to consider negative n?!!
    # actually I think not because we already consider u -> v and v -> u

    # k_dot_r_mu = np.multiply(k_x, x_mu, dtype='float64') + np.multiply(k_y, y_mu, dtype='float64')
    # k_dot_r_nu = np.multiply(k_x, x_nu, dtype='float64') + np.multiply(k_y, y_nu, dtype='float64')
    k_dot_r_mu = k_x * x_mu + k_y * y_mu
    k_dot_r_nu = k_x * x_nu + k_y * y_nu

    # cos_term1 = np.cos(k_dot_r_mu).sum(axis=(0, 1))
    # cos_term2 = np.cos(k_dot_r_nu).sum(axis=(0, 1))
    cos_term1 = common.numba_sum_3d_axis01(np.cos(k_dot_r_mu))
    cos_term2 = common.numba_sum_3d_axis01(np.cos(k_dot_r_nu))
    cos_accum = cos_term1 * cos_term2
    # del cos_term1, cos_term2
    
    # sin_term1 = np.sin(k_dot_r_mu).sum(axis=(0, 1))
    # sin_term2 = np.sin(k_dot_r_nu).sum(axis=(0, 1))
    sin_term1 = common.numba_sum_3d_axis01(np.sin(k_dot_r_mu))
    sin_term2 = common.numba_sum_3d_axis01(np.sin(k_dot_r_nu))
    sin_accum = sin_term1 * sin_term2
    # del sin_term1, sin_term2
    # del k_dot_r_mu, k_dot_r_nu
    
    num_particles = (num_particles_0 + num_particles_1) / 2
    contrib = 1/num_particles * ( cos_accum + sin_accum )
    k = np.sqrt(k_x**2 + k_y**2)
    # del cos_accum, sin_accum # probably unneeded

    return k, contrib

def self_intermediate_scattering_internal(particles_t0, particles_t1, k_x, k_y):
    # Thorneywork et al 2018 eq (27)

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
    # TODO: do we not need to consider negative n?!!
    # actually I think not because we already consider u -> v and v -> u

    # k_dot_dr = k_x * (x_mu - x_nu)  +  k_y * (y_mu - y_nu)
    k_dot_dr = np.multiply(k_x, x_mu - x_nu, dtype='float64') + np.multiply(k_y, y_mu - y_nu, dtype='float64')

    S = 1/num_particles * np.cos(k_dot_dr).sum(axis=(0))
    del k_dot_dr
    #neg = np.sum(contrib < 0)
    #print(f'{neg/contrib.size:.2f} negative')

    k = np.sqrt(k_x**2 + k_y**2)

    return k, S


def individual_exponential_fit_over_t(t, F, k):
    assert t.shape == F.shape
    #t = t[~np.isnan(F)]
    #F = F[~np.isnan(F)]

    F_index_first_order_of_magnitude = np.argmax(F < 0.02)
    if F_index_first_order_of_magnitude == 0: # this means that all F were > 0.3
        F_index_first_order_of_magnitude = F.shape[0] - 1

    fit_end = max(2, F_index_first_order_of_magnitude)
    print('index', F_index_first_order_of_magnitude, 'fit_end', fit_end)
    #print('fit end', fit_end)
    #fit_end=4

    fit_func = lambda t, D : -k.magnitude**2 * t * D
    popt, pcov = scipy.optimize.curve_fit(fit_func, t[:fit_end].magnitude, np.log(F[:fit_end]))
    grad = popt[0]
    grad_unc = np.sqrt(pcov)[0][0]

    #t_fit = np.linspace(t.min(), t.max())
    t_fit = np.logspace(np.log10(0.3), np.log10(t.magnitude.max()))

    F_fit = np.exp(fit_func(t_fit, popt[0])) # exp b/c we logged earlier
    
    D     = popt[0]
    D_unc = pcov[0]

    t_fit = t_fit * t.units

    return D, t_fit, F_fit, fit_end

def individual_exponential_fit_over_k(k, F, t):
    assert k.shape == F.shape
    #k = k[~np.isnan(F)]
    #F = F[~np.isnan(F)]

    F_index_first_order_of_magnitude = np.argmax(F < 0.02)
    if F_index_first_order_of_magnitude == 0: # this means that all F were > 0.3
        F_index_first_order_of_magnitude = F.shape[0] - 1

    fit_end = max(2, F_index_first_order_of_magnitude)
    print('index', F_index_first_order_of_magnitude, 'fit_end', fit_end)
    #print('fit end', fit_end)
    #fit_end=4

    fit_func = lambda k, D : -k**2 * t.magnitude * D
    popt, pcov = scipy.optimize.curve_fit(fit_func, k[:fit_end].magnitude, np.log(F[:fit_end]))

    #t_fit = np.linspace(t.min(), t.max())
    k_fit = np.logspace(np.log10(k.magnitude.min()), np.log10(k.magnitude.max()))

    F_fit = np.exp(fit_func(k_fit, popt[0])) # exp b/c we logged earlier
    
    D     = popt[0]
    D_unc = pcov[0]

    k_fit = k_fit * k.units

    return D, k_fit, F_fit, fit_end


def small_decay_plot(ax, log, x, Fs, x_fit, F_fit, fit_end,
                     show_x_labels=True, show_y_labels=False, title='',
                     hide_colour_diff=False):
    
    excluded_fit_colour = 'tab:blue' if hide_colour_diff else 'cornflowerblue'
    ax.scatter(x[fit_end:], Fs[fit_end:],                      color=excluded_fit_colour)
    ax.scatter(x[:fit_end], Fs[:fit_end], label="observation", color='tab:blue')

    if log:
        ax.semilogy()
        ax.set_xlim(0, x[fit_end]*2)
        ax.set_ylim(0.004, 1.1)
        ax.set_ylim(0.001, 1.1)
        import matplotlib.ticker
        
        # we want to show min and minor (logarithmic) ticks but not numbers
        minor_ticks = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
        ax.yaxis.set_minor_locator(minor_ticks)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(axis='y', which='minor', length=2, direction='out')
        major_ticks = matplotlib.ticker.LogLocator(base=10.0,subs=(1,),numticks=12)
        ax.yaxis.set_major_locator(major_ticks)
        ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(axis='y', which='major', length=4, direction='out')
    else:
        ax.semilogx()
        ax.set_ylim(0, 1.1)

    if not show_x_labels:
        ax.set_xticks([])
        ax.set_xlabel("")
    else:
        ax.set_xlabel("$k (s)$")

    if not show_y_labels:
        ax.set_ylabel("")
        ax.set_yticks([])
    else:
        ax.set_ylabel('$F_s(k, t)$')
    
    if title:
        ax.set_title(title)
    #if np.where(disp_indexes==k_index)[0][0] != 0:
    #t_plot=np.logspace(-1, 3)*units.second
    #ax.plot(t_plot, np.exp(fit_func(t_plot, grad)), label=f"fit : D={D.magnitude:.3f}±{D_unc:.3f~P}", color='tab:orange')
    ax.plot(x_fit, F_fit, color='tab:orange', label='fit')

def D_over_k(ks, Fs, dts, display_k_indexes=[], display_axes=[]):
    num_ks = ks.shape[1]

    Ds     = np.full((num_ks,), np.nan)
    D_uncs = np.full((num_ks,), np.nan)

    for k_index in range(0, num_ks):
        print(k_index, ks.shape)
        k = ks[0, k_index]
        F = Fs[:, k_index]

        if np.all(F == 0):
            # idk but every element of S is zero
            print("skipping, every element of S was zero")
            Ds    [k_index] = np.nan
            D_uncs[k_index] = np.nan
            continue

        # remove negative S and nans
        used_S   = F  [F >= 0 & ~np.isnan(F)]
        used_dts = dts[F >= 0 & ~np.isnan(F)]
        
        if used_S.size == 0:
            # we got only nans ?!!
            print("skipping")
            continue

        D, t_fit, F_fit, fit_end = individual_exponential_fit_over_t(used_dts, used_S, k)

        if k_index in display_k_indexes:
            ax = display_axes[np.argmax(display_k_indexes == k_index)]

            small_decay_plot(ax, True, used_dts, used_S, t_fit, F_fit, fit_end,
                            title = f"$F_s({k.magnitude:.1f}, t)$")


        Ds[k_index] = D

    return ks[0, :], Ds

def D_over_t(dts, Fs, ks, display_t_indexes=[], display_axes=[]):

    # at t=0 the line is flat so u can't fit
    dts = dts[1:]
    ks = ks[1:, :]
    Fs = Fs[1:, :]

    print('dts shape', dts.shape, ks.shape)
    num_ts = dts.shape[0]

    Ds     = np.full((num_ts,), np.nan)
    D_uncs = np.full((num_ts,), np.nan)


    for t_index in range(0, num_ts):
        k = ks [0, :]
        t = dts[t_index]
        F = Fs [t_index, :]

        if np.all(F == 0):
            # idk but every element of S is zero
            print("skipping, every element of S was zero")
            Ds    [t_index] = np.nan
            D_uncs[t_index] = np.nan
            continue

        # remove negative S and nans
        used_S = F[F >= 0 & ~np.isnan(F)]
        used_k = k[F >= 0 & ~np.isnan(F)]
        
        if used_S.size == 0:
            # we got only nans ?!!
            print("skipping")
            continue

        D, k_fit, F_fit, fit_end = individual_exponential_fit_over_k(used_k, used_S, t)

        if t_index in display_t_indexes:
            ax = display_axes[np.argmax(display_t_indexes == t_index)]

            small_decay_plot(ax, True, used_k, used_S, k_fit, F_fit, fit_end,
                            title = f"$F_s(k, {t:.0f~P})$")


        Ds[t_index] = D

    return dts, Ds