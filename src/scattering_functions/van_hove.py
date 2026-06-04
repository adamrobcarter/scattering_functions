import numpy as np
import scipy.stats
import tqdm

def density_correlation(particles_t0, particles_t1, max_r, num_r_bins, crop_border):
    r_bin_edges = np.linspace(0, max_r, num_r_bins+1)
    r_bin_width = r_bin_edges[1] - r_bin_edges[0]
    densities = np.zeros((num_r_bins,))

    assert particles_t0.shape[0] > 0
    assert particles_t1.shape[0] > 0


    x = particles_t0[:, 0]
    y = particles_t0[:, 1]
    width  = max(particles_t0[:, 0].max(), particles_t1[:, 0].max())
    height = max(particles_t0[:, 1].max(), particles_t1[:, 1].max())

    out_of_crop = (x < crop_border) | (x > (width -crop_border)) | (y < crop_border) | (y > (height-crop_border))
    cropped_fraction = out_of_crop.sum() / x.size
    assert cropped_fraction < 1, 'all particles were cropped out'
    # used_locations = np.delete(locations, out_of_crop, axis=0)
    used_locations_t0 = particles_t0[~out_of_crop, :]
    num_used_particles_t0 = used_locations_t0.shape[0]

    # precalculate distance in one go
    used_locations_x_t0 = used_locations_t0[:, 0]
    used_locations_y_t0 = used_locations_t0[:, 1]
    locations_x_t1 = particles_t1[:, 0]
    locations_y_t1 = particles_t1[:, 1]
    dx = used_locations_x_t0[:, np.newaxis] - locations_x_t1[np.newaxis, :]
    dy = used_locations_y_t0[:, np.newaxis] - locations_y_t1[np.newaxis, :]
    r = np.sqrt(dx**2 + dy**2) #  r is num_particles x num_all_particles  where elements are the distance

    # find number of interparticle distances in donuts
    num_particles_in_donuts = np.histogram(r, r_bin_edges)[0]
    num_donuts = num_used_particles_t0
    assert num_donuts > 0
    avg_particles_per_donut = num_particles_in_donuts / num_donuts
    donut_areas = np.pi * (r_bin_edges[1:]**2 - r_bin_edges[:-1]**2) # approx = (left_edge + r_bin_width/2) * r_bin_width
    densities = avg_particles_per_donut / donut_areas

    r = (r_bin_edges[1:] + r_bin_edges[:-1])/2

    avg_density = num_used_particles_t0 / ( (width - 2*crop_border) * (height - 2*crop_border) )
    densities = densities / avg_density
    return r, densities, avg_density


def density_correlation_self(particles_t0, particles_t1, max_r, num_r_bins, crop_border):
    # particles_t0 and particles_t1 have the same shape - they are the same particles at different times
    assert particles_t0.shape == particles_t1.shape

    r_bin_edges = np.linspace(0, max_r, num_r_bins+1)
    r_bin_width = r_bin_edges[1] - r_bin_edges[0]
    densities = np.zeros((num_r_bins,))

    assert particles_t0.shape[0] > 0
    assert particles_t1.shape[0] > 0


    x = particles_t0[:, 0]
    y = particles_t0[:, 1]
    width  = max(particles_t0[:, 0].max(), particles_t1[:, 0].max())
    height = max(particles_t0[:, 1].max(), particles_t1[:, 1].max())

    out_of_crop = (x < crop_border) | (x > (width -crop_border)) | (y < crop_border) | (y > (height-crop_border))
    cropped_fraction = out_of_crop.sum() / x.size
    assert cropped_fraction < 1, 'all particles were cropped out'
    # used_locations = np.delete(locations, out_of_crop, axis=0)
    used_particles = ~out_of_crop
    used_locations_t0 = particles_t0[used_particles, :]
    num_used_particles_t0 = used_locations_t0.shape[0]

    # precalculate distance in one go
    dx = particles_t0[used_particles, 0] - particles_t1[used_particles, 0]
    dy = particles_t0[used_particles, 1] - particles_t1[used_particles, 1]
    r = np.sqrt(dx**2 + dy**2) #

    # used_locations_x_t0 = used_locations_t0[:, 0]
    # used_locations_y_t0 = used_locations_t0[:, 1]
    # locations_x_t1 = particles_t1[:, 0]
    # locations_y_t1 = particles_t1[:, 1]
    # dx = used_locations_x_t0[:, np.newaxis] - locations_x_t1[np.newaxis, :]
    # dy = used_locations_y_t0[:, np.newaxis] - locations_y_t1[np.newaxis, :]
    # r = np.sqrt(dx**2 + dy**2) #  r is num_particles x num_all_particles  where elements are the distance

    # find number of interparticle distances in donuts
    num_particles_in_donuts = np.histogram(r, r_bin_edges)[0]
    num_donuts = num_used_particles_t0
    assert num_donuts > 0
    avg_particles_per_donut = num_particles_in_donuts / num_donuts
    donut_areas = np.pi * (r_bin_edges[1:]**2 - r_bin_edges[:-1]**2) # approx = (left_edge + r_bin_width/2) * r_bin_width
    densities = avg_particles_per_donut / donut_areas

    r = (r_bin_edges[1:] + r_bin_edges[:-1])/2

    avg_density = num_used_particles_t0 / ( (width - 2*crop_border) * (height - 2*crop_border) )
    densities = densities / avg_density
    return r, densities, avg_density

def g_of_r(particles, columns, num_r_bins=20, num_time_origins=20, max_r=None):
    """
    :param particles: array of rows of (x, y, t), possibly with z or id
    :columns: dict with keys 'x', 'y', 't' and values the column index of those in the particles array
    :param num_r_bins: number of radial bins for the g(r) calculation
    :param num_time_origins: number of time frames to average over. computational time is linearly proportional to this
    """
    time_column = columns['t']
    assert columns['x'] == 0
    assert columns['y'] == 1

    if max_r is None:
        max_r = min(particles[:, 0].max(), particles[:, 1].max()) / 2

    num_timesteps = int(particles[:, time_column].max()) + 1
    gs = np.full((num_timesteps, num_r_bins), np.nan)
    
    # fig, ax = plt.subplots(1, 1)
    # ax.set_ylim(0, 2)

    n = num_timesteps # if file in ['eleanor0.01', 'alice0.02'] else 20

    time_origins = [int(i) for i in np.linspace(0, num_timesteps-1, num_time_origins)]

    for time_origin_index, time_origin in enumerate(tqdm.tqdm(time_origins)):
        # t0 = time.time()
        particles_at_t = particles[:, time_column] == time_origin
        assert particles_at_t.sum() > 0
        # t1 = time.time()
        r_bin_edges, g, avg_density = density_correlation(particles[particles_at_t, :], particles[particles_at_t, :], max_r=max_r, num_r_bins=num_r_bins, crop_border=10)
        gs[time_origin_index, :] = g
        # t2 = time.time()

        # a = t1-t0
        # b = t2-t1
        # t = t2-t0
        # print(f'{a/t:.2f}, {b/t:.2f}')

        # ax.clear()
        # ax.set_ylim(0, 3)

        # ax.errorbar(r_bin_edges, np.nanmean(gs, axis=0), yerr=np.nanstd(gs, axis=0)/np.sqrt(n), marker='.', linestyle='none')
        # ax.semilogy()

    return gs, r_bin_edges