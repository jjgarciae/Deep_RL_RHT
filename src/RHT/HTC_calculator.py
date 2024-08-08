import logging
from typing import Optional, Union

import numpy as np
import scipy as sci

from src.utils import set_logging


def HTC_material_defs(
    initial_frequency: float,
    final_frequency: float,
    N_frequencies: int,
    save_path: Optional[str] = "data/HTC",
):
    """Define the HTC function inputs.

    Parameters
    ----------
    initial_frequency : float
        Initial frequency to consider.
    final_frequency : float
        Final frequency to consider.
    N_frequencies : int
        Number of frequences to consider.
    save_path : Optional[str], optional
        Save the result in a .npy file if path is given.
        By default, 'data/HTC'

    Returns
    -------
    mat_per : np.ndarray
        Vector containing the permitivity of each material at each frequency.
        The material we are referring to is indicated through the shape:
        (n_materials, n_freq).
    mat_w : np.ndarray
        Vector containing the frequencies at which the materials permitivity were
        sampled.
    """
    # frecuencies
    wwe = np.linspace(initial_frequency, final_frequency, N_frequencies)

    # calculous of the permitivity of the metal at ecah frecuency
    e_inf = 1
    wp = 2.5
    gamm = 0.01

    vacuum = np.ones((N_frequencies, 1))
    mat_w = []
    for i in range(N_frequencies):
        mat_w.append(e_inf - (wp**2) / (wwe[i] * (wwe[i] + 1j * gamm)))
    mat_w = np.reshape(np.array(mat_w), (N_frequencies, 1))

    # get the permitivity of all materials at each frecuency
    mat_per = np.concatenate((vacuum, mat_w), axis=1)

    # save both results
    if save_path is not None:
        np.save(f"{save_path}/HTC_frequencies.npy", wwe * 10**14)
        np.save(f"{save_path}/HTC_permitivities.npy", mat_per)

    return mat_per, wwe * 10**14


def count_layers(raw_d: np.ndarray, unit_d: int) -> tuple[np.ndarray, np.ndarray]:
    """External function, takes the materials and joins the different width of
    layers.

    Parameters
    ----------
    raw_d : np.ndarray
        Vector of 0s and 1s (more?) describing the material of each width
        element.
    unit_d : int
        Value of each slice of width, nm.

    Returns
    -------
    ddd : np.ndarray
        Vector containing the width of each layer.
    mats : np.ndarray
        Material of each layer (alternating 0 and 1 unless we add more than 2
        options).
    """
    while raw_d.size != 0 and raw_d[-1] == 1:
        raw_d = np.delete(raw_d, -1)

    if raw_d.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    ddd_temp = [unit_d]
    mats_temp = [raw_d[0]]

    for kuy in range(raw_d.shape[0] - 1):
        new_layer = unit_d

        if raw_d[kuy + 1] == raw_d[kuy]:
            ddd_temp[-1] = ddd_temp[-1] + new_layer

        else:
            ddd_temp.append(new_layer)
            mats_temp.append(raw_d[kuy + 1])

    ddd = np.array(ddd_temp)
    mats = np.array(mats_temp)

    return ddd, mats


def thet(TT, wu):
    """External function, defines the mean thermal energy of a mode of frequency
    wu, T derivative.

    Parameters
    ----------
    TT : _type_
        Temperature in SI.
    wu : _type_
        Frequency in SI.
    """
    # Parameters
    h = 6.626 * 10 ** (-34)  # SI
    hbar = h / (2 * np.pi)  # SI
    kB = 1.381 * 10 ** (-23)  # SI

    T_exp = np.exp((hbar * wu) / (kB * TT))

    values = (hbar**2 * wu**2) / (4 * np.pi**2 * kB * TT**2)

    return (values * T_exp) / (T_exp - 1) ** 2


def k_funct(ku, NN, wu, epsss, ddd, gapd):
    """External function, maths of the Scattering Matrix Formalism, k is a
    scalar.

    Parameters
    ----------
    ku : _type_
        k value for the calculations.
    NN : _type_
        Number of material layers, the full vector has 2 extra components.
    wu : _type_
        Frequency for the calculations.
    epsss : _type_
        Material parameter vector, 2 components longer than NN.
    ddd : _type_
        Widths parameters, one extra 0 on each side.
    gapd : _type_
        Separation between the systems.

    Returns
    -------
    ktau : _type_
        Transmission times k value.
    """
    # Parameters
    c = 3 * 10**8

    # Initialization
    fn = 0
    fn1 = 0
    D = 0

    I11 = np.zeros((2, 2))
    I12 = np.zeros((2, 2))
    I21 = np.zeros((2, 2))
    I22 = np.zeros((2, 2))

    S011 = np.eye(2)
    S012 = np.zeros((2, 2))
    S021 = np.zeros((2, 2))
    S022 = np.eye(2)

    S111 = np.zeros((2, 2))
    S112 = np.zeros((2, 2))
    S121 = np.zeros((2, 2))
    S122 = np.zeros((2, 2))

    wu = wu * 10**14
    ku = ku * 10**14

    # Main loop
    for n in range(NN + 1):
        qn = np.sqrt(epsss[n] * wu**2 - ku**2 + 0j)
        qn1 = np.sqrt(epsss[n + 1] * wu**2 - ku**2 + 0j)

        if np.imag(qn) < 0:
            qn = qn * (-1)

        if np.imag(qn + 1) < 0:
            qn1 = qn1 * (-1)

        D = qn / qn1

        fn = np.exp(1j * qn / c * ddd[n] * 10 ** (-9))
        fn1 = np.exp(1j * qn1 / c * ddd[n + 1] * 10 ** (-9))

        I11 = np.array([[D + 1, 0.0], [0.0, epsss[n] / (D * epsss[n + 1]) + 1]]) / 2
        I12 = np.array([[1 - D, 0.0], [0.0, 1 - epsss[n] / (D * epsss[n + 1])]]) / 2
        I21 = I12
        I22 = I11

        S111 = fn * np.matmul(S011, np.linalg.inv(I11 - fn * np.matmul(S012, I21)))
        S121 = np.matmul(S022, np.matmul(I21, S111)) + S021
        S112 = np.matmul(
            (fn * np.matmul(S012, I22) - I12) * fn1,
            np.linalg.inv(I11 - fn * np.matmul(S012, I21)),
        )
        S122 = np.matmul(S022, (np.matmul(I21, S112) + I22 * fn1))

        S011 = S111
        S012 = S112
        S021 = S121
        S022 = S122

    rp = S121[1, 1] * (-1)
    Rp = np.abs(S121[1, 1]) ** 2
    rs = S121[0, 0] * (-1)
    Rs = np.abs(S121[0, 0]) ** 2

    qinter = np.sqrt(wu**2 - ku**2 + 0j)
    Ds = np.abs(1 - rs * rs * np.exp(2 * 1j * qinter / c * gapd * 10 ** (-9))) ** 2
    Dp = np.abs(1 - rp * rp * np.exp(2 * 1j * qinter / c * gapd * 10 ** (-9))) ** 2

    if ku < wu:
        taus = (1 - Rs) ** 2 / Ds
        taup = (1 - Rp) ** 2 / Dp

    else:
        taus = (
            4
            * np.imag(rs) ** 2
            * np.exp(-2 * np.abs(qinter) / c * gapd * 10 ** (-9))
            / Ds
        )
        taup = (
            4
            * np.imag(rp) ** 2
            * np.exp(-2 * np.abs(qinter) / c * gapd * 10 ** (-9))
            / Dp
        )

    ktau = (taus + taup) * ku

    return ktau


# MAIN
def HTC(
    htc_material: np.ndarray,
    mat_per: np.ndarray,
    mat_w: np.ndarray,
    gap_nm: Union[int, float] = 10,
    unit_width_nm: int = 5,
    logging_level: str = "info",
):
    """External function: in the material of each slice, out the total HTC.

    Parameters
    ----------
    htc_material : np.ndarray
        Vector of 0s and 1s (or more) indicating the material of each layer.
    mat_per : np.ndarray
        Vector containing the permitivity of each material at each frequency.
        The material we are referring to is indicated through the shape:
        (n_materials, n_freq).
    mat_w : np.ndarray
        Vector containing the frequencies at which the materials permitivity were
        sampled.
    gap_nm : Union[int, float], optional
        Size of the gap, nm.
        By default, 10.
    unit_width_nm : int, optional
        Width of each slice of layer, in nm.
        By default, 5 nm.
    logging_level : str, optional
        Select logging level.
        By default, "info".

    Returns
    -------
    htc : float
        Full heat transfer coefficient, single number.
    """
    # set logging level
    set_logging(level=logging_level)

    logging.debug(f"Computing htc of state {htc_material}...")

    # Parameters
    To = 300  # T in K
    8.85 * 10 ** (-12)  # SI
    c = 3 * 10**8  # SI
    h = 6.626 * 10 ** (-34)  # SI
    h / (2 * np.pi)  # SI
    1.381 * 10 ** (-23)  # SI

    Nw = 200  # Number of frequency points
    win = 0.3 * 10**14  # Starting frequency rad/s
    wend = 3.0001 * 10**14  # Ending frequency rad/s
    N_mats = mat_per.shape[1]

    # Initialization
    # Width of each of the non-infinite layer (except last one), material for
    # each of them
    lay_d, mat_d = count_layers(
        np.concatenate((np.array(1, ndmin=1), htc_material)), unit_width_nm
    )
    Nd = lay_d.shape[0]  # Number of non-infinite layers
    ww = np.linspace(win, wend, num=Nw)  # Vector of frequencies

    # Main calculations
    ddd = np.concatenate(
        (np.array(0.0, ndmin=1), lay_d, np.array(0.0, ndmin=1))
    )  # Full vector of widths for the calculations

    mats_temp = []  # Preallocate the mat_char

    for i in range(N_mats):
        mats_temp.append(np.interp(ww, mat_w, mat_per[:, i]))

    mats_temp = np.array(mats_temp)

    eps_mat = np.reshape(mats_temp[0], (Nw, 1))

    for i in range(mat_d.shape[0]):
        eps_mat = np.concatenate(
            (eps_mat, np.reshape(mats_temp[mat_d[i]], (Nw, 1))), axis=1
        )

    eps_mat = np.concatenate(
        (eps_mat, np.reshape(mats_temp[1], (Nw, 1))), axis=1
    )  # Full vector of materials

    res_w = np.zeros(Nw)  # Preallocate the h_w

    for i in range(Nw):
        epsss = eps_mat[i, :]

        part = sci.integrate.quad(
            lambda x: k_funct(x, Nd, ww[i] / 10**14, epsss, ddd, gap_nm),
            0,
            np.inf,
            limit=10000,
            epsrel=10 ** (-4),
        )  # Integrate over k

        res_w[i] = thet(To, ww[i]) * part[0] * 10**14 / c**2

    htc = np.trapz(res_w, ww)

    return htc


if __name__ == "__main__":
    # get HTC inputs
    materials, wwe = HTC_material_defs(0.3, 3, 200)
    # get the htc calculous for the given multilayer configuration
    htc_material = np.array([0, 0, 1, 0, 1])
    htc = HTC(htc_material, materials, wwe)
    print(htc)
