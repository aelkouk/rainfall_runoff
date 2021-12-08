
# Purpose: Produce synthetic precipitation forcing at varying time
# Record of revisions:
# Date      Programmer      Description of change
# ========  =============   =====================
# 09-2020   A. Elkouk       Original code

import numpy as np

def synthetic_precipitation(t, precip_max=100.0, t_peak=2.0, sigma_t=0.24):
    """ Produce synthetic precipitation forcing at time t

    Parameters
    ----------
    t : array_like
        Time [days]
    precip_max : int or float
        Maximum precipitation rate [mm day^-1]
    t_peak : float
        Time of the precipitation peak [days]
    sigma_t : float
        Scale factor affecting the duration precipitation peak

    Returns
    -------
    precip_t : float
        Precipitation amount at time t [mm days^-1]
    """

    precip_t = precip_max * np.exp(-((t_peak - t) / sigma_t) ** 2)
    return precip_t