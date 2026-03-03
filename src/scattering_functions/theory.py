import numpy as np
import scipy.special

def structure_factor_hard_spheres_2d(k, phi, sigma):
    assert np.all(k != None)
    assert phi != None
    assert sigma != None
    assert np.isfinite(phi)
    assert np.isfinite(sigma)
    assert np.isfinite(k).any(), 'all k were nan/inf'
    assert 0 < phi < 1

    S = np.full_like(k, np.nan)

    # sigma is disk diameter
    rho = 4 * phi / (np.pi * sigma**2)

    phi = np.pi/4 * rho * sigma**2
    J0 = lambda x: scipy.special.jv(0, x)
    J1 = lambda x: scipy.special.jv(1, x)

    # we have to calculate for k!=0 and k==0 separately
    prefactor = np.pi / ( 6 * ( 1 - phi)**3 * k[k!=0]**2 )
    line1 = -5/4 * (1 - phi)**2 * k[k!=0]**2 * sigma**2 * J0(k[k!=0] * sigma / 2)**2
    line23 = 4 * ( (phi - 20) * phi + 7) + 5/4 * (1 - phi)**2 * k[k!=0]**2 * sigma**2
    line23factor = J1(k[k!=0] * sigma / 2)**2
    line4 = 2 * (phi - 13) * (1 - phi) * k[k!=0] * sigma * J1(k[k!=0] * sigma / 2) * J0(k[k!=0] * sigma / 2)
    c = prefactor * (line1 + line23*line23factor + line4)
    # ^^^ Thorneywork et al 2018
    
    S[k!=0] = 1 / (1 - rho * c) # Hansen & McDonald (3.6.10)

    # annoying thing to make it handle scalar inputs (maybe outdated probably could be improved)
    if np.isscalar(k):
        if k == 0:
            S = (1 - phi)**3 / (1 + phi)
    else:
        S[k == 0] = (1 - phi)**3 / (1 + phi) 
    return S