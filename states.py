
# Purpose: Calculate coupled and uncoupled states of the canopy, unsaturated and saturated sub-domains
# Record of revisions:
# Date      Programmer      Description of change
# ========  =============   =====================
# 09-2020   A. Elkouk       Original code


from flux_param import (calc_canopy_drainage_flux, calc_canopy_evaporation,
                        calc_overland_flow, calc_percolation_flux,
                        calc_precipitation_excess, calc_saturated_fraction,
                        calc_throughfall_flux, calc_unsaturated_evaporation,
                        calc_wetted_fraction, calc_baseflow)


# ----------------------------------------------------------------------------------------------------------------------
# Coupled States
# ----------------------------------------------------------------------------------------------------------------------

def run_coupled_states(storage, precip, pet, canopyStore_max, unsatStore_max,
                       fieldCap, gamma, alpha, beta, k_can, k_sat, k_sz):
    """ Run the coupled model for vegetation canopy, the unsaturated zone, and the saturated zone

    Parameters
    ----------
    storage : list
        Storage [mm] in each model sub-domain e.i. [canopy, unsaturated, saturated]
    precip : float
        Precipitation flux [mm day^-1]
    pet : float
        Potential evapotranspiration [mm day^-1]
    canopyStore_max : float
        Maximum non-drainable canopy interception storage [mm]
    unsatStore_max : float
        Maximum storage in the unsaturated zone [mm]
    fieldCap : float
        Field capacity [mm]
    gamma : float
        Parameter to account for the non-linearity in the wetted fraction of the canopy
    alpha : float
        Parameter to account for the non-linearity in the variable source area for saturation-excess runoff
    beta : float
        Parameter to account for percolation non-linearity
    k_can : float
        Canopy drainage coefficient [day^-1]
    k_sat : float
        Maximum percolation rate [mm day^-1]
    k_sz : float
        Runoff coefficient for the saturated zone [day^-1]

    Returns
    -------
    states : list
        Storage [mm] in each model sub-domain
    totalRunoff : float
        Total runoff [mm day^-1]
    """

    canopyStore, unsatStore, satStore = storage

    wetFrac = calc_wetted_fraction(canopyStore, canopyStore_max, gamma)
    canopyEvap = calc_canopy_evaporation(pet, wetFrac)
    throughfall = calc_throughfall_flux(precip, canopyStore, canopyStore_max)
    canopyDrain = calc_canopy_drainage_flux(canopyStore, canopyStore_max, k_can)
    canopyStore = precip - (canopyEvap + throughfall + canopyDrain)

    precipExcess = calc_precipitation_excess(throughfall, canopyDrain)
    unsatEvap = calc_unsaturated_evaporation(pet, unsatStore, fieldCap, wetFrac)
    satFrac = calc_saturated_fraction(unsatStore, unsatStore_max, alpha)
    overlandFlow = calc_overland_flow(precipExcess, satFrac)
    percolation = calc_percolation_flux(unsatStore, unsatStore_max, fieldCap, k_sat, beta)
    unsatStore = precipExcess - (unsatEvap + overlandFlow + percolation)

    baseflow = calc_baseflow(satStore, k_sz)
    satStore = percolation - baseflow

    states = [canopyStore, unsatStore, satStore]
    totalRunoff = overlandFlow + baseflow

    return [states, totalRunoff]


# ----------------------------------------------------------------------------------------------------------------------
# Uncoupled States
# ----------------------------------------------------------------------------------------------------------------------

def calc_canopy_state(canopyStore, precip, pet, canopyStore_max, gamma, k_can):
    """ Calculate the state of the vegetation canopy sub-domain

    Parameters
    ----------
    canopyStore : float
        Canopy Interception storage [mm]
    precip : float
        Precipitation flux [mm day^-1]
    pet : float
        Potential evapotranspiration [mm day^-1]
    canopyStore_max : float
        Maximum non-drainable canopy interception storage [mm]
    gamma : float
        Parameter to account for the non-linearity in the wetted fraction of the canopy
    k_can : float
        Canopy drainage coefficient [day^-1]

    Returns
    -------
    canopyStore : float
        Canopy Interception storage [mm]
    """

    wetFrac = calc_wetted_fraction(canopyStore, canopyStore_max, gamma)
    canopyEvap = calc_canopy_evaporation(pet, wetFrac)
    throughfall = calc_throughfall_flux(precip, canopyStore, canopyStore_max)
    canopyDrain = calc_canopy_drainage_flux(canopyStore, canopyStore_max, k_can)
    canopyStore = precip - (canopyEvap + throughfall + canopyDrain)

    return canopyStore


def calc_unsaturated_state(unsatStore, precipExcess, pet, wetFrac, unsatStore_max, fieldCap, alpha, beta, k_sat):
    """ Calculate the state of the unsaturated zone sub-domain

    Parameters
    ----------
    unsatStore : float
        Storage in the unsaturated zone [mm]
    precipExcess : float
        Excess precipitation [mm day^-1]
    pet : float
        Potential evapotranspiration [mm day^-1]
    wetFrac : float
        Wetted fraction of the canopy
    unsatStore_max : float
        Maximum storage in the unsaturated zone [mm]
    fieldCap : float
        Field capacity [mm]
    alpha : float
        Parameter to account for the non-linearity in the variable source area for saturation-excess runoff
    beta : float
        Parameter to account for percolation non-linearity
    k_sat : float
        Maximum percolation rate [mm day^-1]

    Returns
    -------
    unsatStore : float
        Storage in the unsaturated zone [mm]
    """

    satFrac = calc_saturated_fraction(unsatStore, unsatStore_max, alpha)
    unsatEvap = calc_unsaturated_evaporation(pet, unsatStore, fieldCap, wetFrac)
    overlandFlow = calc_overland_flow(precipExcess, satFrac)
    percolation = calc_percolation_flux(unsatStore, unsatStore_max, fieldCap, k_sat, beta)
    unsatStore = precipExcess - (unsatEvap + overlandFlow + percolation)

    return unsatStore


def calc_saturated_state(satStore, percolation, k_sz):
    """ Calculate the state of the saturated zone sub-domain

    Parameters
    ----------
    satStore : int or float
        Storage in the saturated zone [mm]
    percolation : int or float
        Percolation flux [mm day^-1]
    k_sz : float
        Runoff coefficient for the saturated zone [day^-1]

    Returns
    -------
    satStore : int or float
        Storage in the saturated zone [mm]
    """

    baseflow = calc_baseflow(satStore, k_sz)
    satStore = percolation - baseflow

    return satStore
