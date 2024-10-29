import numpy as np
import tqdm
import atooms.trajectory
import atooms.postprocessing
import atooms.system

# help(atooms.postprocessing.fourierspace.FourierSpaceCorrelation)

def intermediate_scattering(log, F_type, crop, num_k_bins, num_iters, d_frames, data, num_frames, max_K, width, height):
    trajs_np = reshape(data)
    num_particles = trajs_np.shape[0]
    num_timesteps = trajs_np.shape[1]
    
    # see https://atooms.frama.io/postprocessing/tutorial/#trajectories-as-numpy-arrays
    with atooms.trajectory.TrajectoryRam() as trajs:
        box = np.ndarray(2)

        for particle in range(num_particles):
            s = atooms.system.System(N=num_timesteps, d=2)
            # s.cell.side[:] = box
            s.view('position')[:, :] = trajs_np[particle, :, :]
            trajs.append(s)
        print('step', trajs.timestep)

        isf_calc = atooms.postprocessing.IntermediateScattering(trajs, tgrid=d_frames, kgrid=[0.1, 0.14, 0.5, 1.3, 2, 4, 8])
        isf_calc.compute()
        print(isf_calc)
        # print(isf_calc.grid.shape)
        # print(isf_calc.value.shape)
        k = np.array(isf_calc.grid[0])
        t = np.array(isf_calc.grid[1])
        print(f'atooms outputted k.size = {k.size}, t.size = {t.size}')

        print(isf_calc.skip)
        print(d_frames)
        print(t)
        print(d_frames.size, t.size)
        # assert d_frames.size == t.size

        F = np.array(isf_calc.value)
        print(len(k), len(t), len(F), len(F[0]))
        
        print('k', k)

        k = np.tile(k, (F.shape[1], 1)) # k needs to be a grid over F values

        F = F.transpose() # this is to match the format from before

        return F, np.zeros_like(F), k, t

def reshape(all_data):
    # reformat to be (num particles) x (num timesteps) x (x, y)
    if all_data.shape[1] == 4:
        num_particles = int(all_data[:, 3].max())
        num_timesteps = int(all_data[:, 2].max())

        print(all_data[:, 3].max(), all_data[:, 3].min())
        print(all_data[:, 2].max(), all_data[:, 2].min())

        data = np.full((num_particles, num_timesteps, 2), np.nan)

        for row_index in tqdm.trange(all_data.shape[0]):
            x, y, t, num = all_data[row_index]
            data[int(num)-1, int(t)-1, :] = x, y # -1 cause they are 1-based

        # remove any nans?
        to_remove = []
        for particle in range(num_particles):
            if np.any(np.isnan(data[particle, :, :])):
                to_remove.append(particle)
        data = np.delete(data, to_remove, axis=0)
        print(f'removed {len(to_remove)/num_particles*100:.1f}%')

        return data

        width  = ( np.nanmax(data[:, :, 0]) - np.nanmin(data[:, :, 0]) )
        height = ( np.nanmax(data[:, :, 1]) - np.nanmin(data[:, :, 1]) )

    if all_data.shape[1] == 3:
        # no particle id provided

        max_t = int(all_data[:, 2].max()+1)
        data = [[] for _ in range(max_t)]
        max_sim = 0

        for row_index in tqdm.trange(all_data.shape[0]):
            x, y, t = all_data[row_index]
            t = int(t)
            
            # if t >= len(data):
            #     data.append([])

            data[t].append([x, y])
            max_sim = max(max_sim, len(data[t]))

        real_data = np.full((max_sim, max_t, 2), np.nan)
        for t in range(max_t):
            print(t, data[t])
            real_data[:len(data[t]), t, :] = data[t]

        real_data = np.array(real_data)
        print(real_data.shape)
        print(real_data)
        return real_data