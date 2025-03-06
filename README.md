# Scattering Functions

A library to calculate the intermediate scattering function / dynamic structure factor $F(k, t)$ from 2D particle coordinates, obtained from (eg) TrackPy. Can therefore also calulate the structure factor $S(k)$. Ability to calculate the self-ISF $F_s(k, t)$ is coming.

## Instalation
Clone the repository somewhere, and run `pip install -e .`. Dependencies are numpy and scipy. If tqdm is available, we will use it for pretty progress bars.

## Use
Now you should be able to do `import scattering_functions` in python. See `example.py` for more.

## Details
The code calculates
```math
F(k, t) = \frac1N \sum_\mu \sum_\nu \langle \exp(-i\vb{k} \cdot [\vb{r_\mu}(t + t_0) - \vb{r_\nu}(t_0)]) \rangle
```
where $\vb{r_\mu}(t)$ are particle coordinates at time $t$ and the angle brackets indicate an average over \textit{time origins} $t_0$

The main entry point, `scattering_functions.intermediate_scattering` has the following parameters

| Parameter          | Default | Description |
| ------------------ | ------- | ----------- |
| F_type             |         | use `'F'` to calculate $F(k, t)$ |
| min_K              |         | a length-2 array like object of the minimum k value to calculate in each dimension. This should probably be $(2\pi / L_x, 2\pi / L_y)$ where $L_x$ and $L_y$ are the sizes of the observation window |
| max_K              |         | the maximum k value to calculate to |
| num_k_bins         | 50      | the returned $k$ points will be logarithmically spaced between `min_K` and `max_K` with this many points. The logarithmically spaced points will be quantised to the nearest multiple of `min_K`. Note that this may result in fewer points being returned than requested, if two logarithmically spaced points get quantised to the same multiple of `min_K` |
| use_doublesided_k  | False   | The section of $k$-space for $k_y < 0$ is redundant as it's an exact copy of the $k_y > 0$ space rotated around the origin. If set to true, calculate over the entire $k$-space anyway |
| d_frames           |         | The
| cores              |         | if >1, the computation will run with the selected number of cores (via multiprocessing) |
| particles_at_frame |         | the particle coordinates - see `get_particles_at_frame` below |
| times              |         | a 1D array of timesteps in the data - see `get_particles_at_frame` below |
| max_time_origins   |         | maximum number of time origins to average over. The actual number of time origins averaged over may be lower if there are fewer frames with time difference $d_frame[i]$ in the dataset |

`scattering_functions.intermediate_scattering` requires the particle coordinates to be in a specific format - an array of shape (number of timesteps) x (max number of particles per frame) x (x, y). `scattering_functions.get_particles_at_frame` converts an array of rows of (x, y, t) coordinates (as output from TrackPy) to this format. It has parameters `F_type` which should be `'F'` for calculating $F(k, t)$, and `particles`, which should be the TrackPy output array. The time coordinate of the `particles` array does not have to be integer nor evenly spaced. It returns `particles_at_frame` and  `times` for use with `scattering_functions.intermediate_scattering`.

`scattering_functions.intermediate_scattering` returns an object with the following properties

| Property   | Description |
| ---------- | ----------- |
| F          | $F(k, t)$, with the radial average over $k$ vectors with the same magnitude taken. Array of shape len(d_frames) x (num k points) |
| F_unc      | uncertainty on `F`. It is the standard error over the different time origins used for each $t$ point. Same shape as `F` |
| k          | $k$ values, with the same shape as `F`. Note that every row is therefore identical |
| F_full     | this is $F(k, t)$ before the radial average is taken. Array of shape len(d_frames) x (num k points x) x (num k points y) |
| F_unc_full | uncertainty on `F_full`. Same shape as `F_full` |
| k_full     | $k$ values for `F_full`. Same shape as `F_full` |
| k_x        | $k$ values used to create the grid in the x direction |
| k_y        | $k$ values used to create the grid in the y direction |


## Support
This code is very much pre-release and is liable to change at any time. If the code would be helpful to you, feel free to contact me (adam.rob.carter@gmail.com) while the documentation is lacking.