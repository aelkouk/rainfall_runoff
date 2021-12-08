# Purpose: Implement time stepping scheme to solve individual and coupled state equations
# Record of revisions:
# Date      Programmer      Description of change
# ========  =============   =====================
# 09-2020   A. Elkouk       Original code

import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Explicit (forward) Euler method
# ----------------------------------------------------------------------------------------------------------------------

def explicitEuler(f_S, S0, T, dt, **kwargs):
    """ Solve dS/dt = f_S(S, t), S(0)=S0, for n=T/dt steps

    Parameters
    ----------
    f_S : function
        State function for given model sub-domain (canopy, unsaturated zone, saturated zone)
    S0 : float
        State initial condition at t=0
    T : float or int
        Time period [days]
    dt : float or int
        Time step [days]
    kwargs : dict
        *kwargs* are used to specify the additional parameters used by the the state function (f_S)

    Returns
    -------
    dS : ndarray
        Integrated state for n=T/dt time steps
    t : ndarray
        Time steps [days]
    """
    n = int(T / dt)
    t = np.zeros(n + 1)
    dS = np.zeros(n + 1)
    dS[0] = S0
    t[0] = 0
    for k in range(n):
        t[k + 1] = t[k] + dt
        dS[k + 1] = dS[k] + (dt * f_S(dS[k], **kwargs))

        if dS[k + 1] < 0.0:
            dS[k + 1] = 0.0
    return dS, t


def explicitEuler_coupled_states(f_S, S0, T, dt, precip, **kwargs):
    """ Solve dS/dt = f_S(S, t), S(0)=S0, for n=T/dt steps

    Parameters
    ----------
    f_S : function
        Coupled state function of the model sub-domains (e.g. canopy, unsaturated zone, and saturated zone)
    S0 : array_like
        State initial conditions at t=0 (e.g. [canopy, unsaturated zone, and saturated zone])
    T : float or int
        Time period [days]
    dt : float or int
        Time step [days]
    precip : array_like
        Precipitation flux [mm days^-1] at n=T/dt time step
    kwargs : dict
        *kwargs* are used to specify the additional parameters used by the the state function (f_S)

    Returns
    -------
    dS : ndarray
        Integrated states for n=T/dt time steps with shape (n, nbr_states)
    RO : ndarray
        Total runoff [mm day^-1] for n=T/dt time steps
    t : ndarray
        Time steps [days]
    """
    n = int(T / dt)
    nS = len(S0)
    t = np.zeros(n + 1)
    dS = np.zeros((n + 1, nS))
    RO = np.zeros(n + 1)
    dS[0, :] = S0
    t[0] = 0
    for k in range(n):
        t[k + 1] = t[k] + dt
        Sk, RO_k = f_S(dS[k], precip[k], **kwargs)
        RO[k] = RO_k
        dS[k + 1, :] = dS[k, :] + (dt * np.array(Sk))
        dS = np.where(dS < 0.0, 0.0, dS)

    return dS, RO, t


# ----------------------------------------------------------------------------------------------------------------------
# Heun's method
# ----------------------------------------------------------------------------------------------------------------------

def Heun(f_S, S0, T, dt, **kwargs):
    """ Solve dS/dt = f_S(S, t), S(0)=S0, for n=T/dt steps using the explicit Heun's method

    Parameters
    ----------
    f_S : function
        State function for given model sub-domain (canopy, unsaturated zone, saturated zone)
    S0 : float
        State initial condition at t=0
    T : float or int
        Time period [days]
    dt : float or int
        Time step [days]
    kwargs : dict
        *kwargs* are used to specify the additional parameters used by the the state function (f_S)

    Returns
    -------
    dS : ndarray
        Integrated state for n=T/dt time steps
    t : ndarray
        Time steps [days]
    """

    n = int(T / dt)
    t = np.zeros(n + 1)
    dS = np.zeros(n + 1)
    dS[0] = S0
    t[0] = 0
    for k in range(n):
        t[k + 1] = t[k] + dt
        K1 = f_S(dS[k], **kwargs)
        K2 = f_S((K1 * dt) + dS[k], **kwargs)
        dS[k + 1] = dS[k] + (0.5 * dt * (K1 + K2))
        if dS[k + 1] < 0.0:
            dS[k + 1] = 0.0
    return dS, t


def Heun_ndt(f_S, Si, dt, T, **kwargs):
    """ Solve dS/dt = f_S(S, t), S(t=t_i)=Si, at t=T with n steps (n=T/dt), using the explicit Heun's method

    Parameters
    ----------
    f_S : function
        State function for given model sub-domain (canopy, unsaturated zone, saturated zone)
    Si : float
        State at time t=t_i
    dt : float or int
        Time step [days]
    T : float or int
        Time period [days]
    kwargs : dict
        *kwargs* are used to specify the additional parameters used by the the state function (f_S)

    Returns
    -------
    dS : float
        Integrated state at t=ndt with n=T/dt
    """

    n = int(T / dt)
    dS = Si
    for _ in range(n):
        K1 = f_S(dS, **kwargs)
        K2 = f_S((K1 * dt) + dS, **kwargs)
        dS = dS + (0.5 * dt * (K1 + K2))
    return dS


def Heun_adaptive_substep(f_S, Si, dt, T, tau_r, tau_abs, s=0.9, rmin=0.1, rmax=4.0, EPS=10 ** (-10), **kwargs):
    """ Solve dS/dt = f_S(S, t), S(t=t_i)=Si, using the explicit Heun's method with numerical
    error control and adaptive sub stepping

    Parameters
    ----------
    f_S : function
        State function for given model sub-domain (canopy, unsaturated zone, saturated zone)
    Si : float
        State at time t=t_i
    dt : float or int
        Time step [days]
    T : float or int
        Time period [days]
    tau_r : float
        Relative truncation error tolerance
    tau_abs : float
        Absolute truncation error tolerance
    s : float
        Safety factor
    rmin, rmax : float
        Step size multiplier constraints
    EPS : float
        Machine constant
    kwargs : dict
        *kwargs* are used to specify the additional parameters used by the the state function (f_S)

    Returns
    -------
    dS : list
        Integrated state
    time : list
        Time steps [days]
    """
    t = 0
    dS = [Si]
    time = [t]
    while t < T:
        t += dt
        y1 = Heun_ndt(f_S, Si, dt, dt, **kwargs)
        y2 = Heun_ndt(f_S, Si, dt/2, dt, **kwargs)
        err = abs(y1 - y2)
        diff = err - ((tau_r * abs(y2)) + tau_abs)
        if diff < 0:
            Si = y2
            dS.append(Si)
            time.append(dt)
            dt = dt * min(s * np.sqrt((tau_r * abs(y2) + tau_abs) / (max(err, EPS))), rmax)
        elif diff > 0:
            t -= dt
            dt = dt * max(s * np.sqrt((tau_r * abs(y2) + tau_abs) / (max(err, EPS))), rmin)
    return dS, time
