import itertools
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from global_vars import WORKING_DIR

# for HPC, specify the path to import from our modules
# .. [1] https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
sys.path.insert(1, WORKING_DIR)  # [1]

# import the rest of all necessary libraries
from src.RHT.HTC_calculator import HTC  # noqa E402
from src.RL.plots import learning_curve  # noqa E402
from src.utils import (
    exists,
    load_conf,
    load_hdf_to_series,
    load_or_update_series,
    set_logging,
    store_series_to_hdf,
    str_to_tuple_or_list,
    timer,
    tuple_or_list_to_str,
)


def htc_dummy_simulation(
    subset_size: Optional[int] = None,
    cfg_path: Path = Path("./src/config.toml"),
    save_computed_htc: bool = False,
    plot_learning_curves: bool = False,
    logging_level: str = "warn",
    seed: Optional[int] = None,
) -> tuple[learning_curve, pd.Series]:
    """Obtain the "optimal state" obtained with Reinforcement Learning for the
    Radiative Heat Transfer simulation naively, this is, compute the
    Heat-Transfer Coefficient (HTC) of a subset of all states and
    select the state wich returns the highest HTC.

    Parameters
    ----------
    subset_size : Optional[int], optional
        Maximum number of states to consider. They will be randomly sampled.
        If None, sample all possible states. By default, None.
    cfg_path : Path, optional
        Config path.
        By default, Path("./src/config.toml")
    save_computed_htc : bool, optional
        Overwrite stored htc values in order to add new computed values. If
        several runs are executed at the same time, this could lead to execution
        failure.
        By default, False.
    plot_learning_curves : bool, optional
        Select to plot `best htc` curve or not.
        By defaul, False
    logging_level : str, optional
        Select logging level.
        By default, `warn`.
    seed : int, optional
        Seed number for random module. If None, do not fix any seed.
        By default, None.

    Return
    ------
    htc_curve : learning_curve
        Object with `htc_curve` info.
    states_htc_series : pd.Series
        Series with all explored state-htc pairs. States as index, htc as
        values.

    Warnings
    --------
    `HTC_{n_layers}_layers.h5` input can be problematic.
        If it has been previously **UNsuccessfully** generated, a HDF5ExtError
        will raise.
        To solve it, delete `HTC_{n_layers}_layers.h5` and execute twice
        the simulation, as the first modification to the file will raise a
        ValueError as file is already opened and it is required to be in
        read-only mode.

    Warns
    -----
    * Remember this function is intended for htc problem. A careful review must
      be done before employing it for further purposes.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/45156630/exploding-memory-usuing-permutations-from-itertools
    .. [2] https://stackoverflow.com/questions/5176232/python-returning-itertools-combinations-object-at-0x10049b470-how-can-i-ac
    .. [3] https://stackoverflow.com/questions/52588298/pandas-idxmax-return-all-rows-in-case-of-ties
    """
    # fix seed
    if exists(seed):
        random.seed(seed)

    # set logging level
    set_logging(level=logging_level)

    # load the configuration
    cgf_rht = load_conf(cfg_path, key="RHT")["environment"]
    cfg_dummy = load_conf(cfg_path, key="dummy")

    saving_frequency = cfg_dummy.get("saving_frequency", 1)

    # create plot object
    htc_curve = learning_curve()

    # store start time of the program
    start_time = time.time()

    # load HTC input constants
    # frequencies range
    with open("./data/HTC/HTC_frequencies.npy", "rb") as f:
        freq_range = np.load(f)
    # materials permitivities at each frequencies
    with open("./data/HTC/HTC_permitivities.npy", "rb") as f:
        mat_per = np.load(f)

    # get all possible states
    if subset_size is None:
        tuple_state_space = list(
            itertools.product(cgf_rht["materials"], repeat=cgf_rht["n_layers"])
        )

    # get a random subset of all possible states [1]
    else:
        # for htc problem, the number of possible states is given by the number
        # of materials powered to the number of layers
        full_n_states = pow(len(cgf_rht["materials"]), cgf_rht["n_layers"])
        # select random indexes of different states
        indexes = random.sample(range(full_n_states), subset_size)

        # retrieve the states corresponding to previous randomly chosen indexes
        tuple_state_space = [
            list(  # itertools returns an iterator, thus it must be called [2]
                itertools.islice(
                    itertools.product(cgf_rht["materials"], repeat=cgf_rht["n_layers"]),
                    index,
                    index + 1,
                    1,
                )
            )[0]
            for index in indexes
        ]
    state_space = tuple_or_list_to_str(tuple_state_space)

    # obtain the htc from indicated path or calculate, store and employ it
    if exists(cgf_rht["htc_path"]):
        htc_values = load_hdf_to_series(
            f"{cgf_rht['htc_path']}/HTC_{cgf_rht['n_layers']}_layers.h5",
            all_index=state_space,
        )

    # initialize some elements
    max_htc = 0
    states_htc_series = pd.Series()

    # randomize state_space exploration
    random.shuffle(state_space)

    # retrieve some htc values
    for idx, state in enumerate(state_space):

        # index + 1 so we start counting from 1
        state_idx = idx + 1

        # output state number
        logging.debug(f"State {state_idx}/{len(state_space)}")

        if cgf_rht["htc_path"] is None:
            htc = HTC(
                htc_material=str_to_tuple_or_list(state, to="list"),
                mat_per=mat_per,
                mat_w=freq_range,
                gap_nm=cgf_rht["gap_nm"],
                unit_width_nm=cgf_rht["layer_width_nm"],
                logging_level=logging_level,
            )
        else:
            htc, htc_values = load_or_update_series(
                htc_values,
                desired_index=state,
                function=HTC,
                empty_value=np.nan,
                htc_material=str_to_tuple_or_list(state, to="list"),
                mat_per=mat_per,
                mat_w=freq_range,
                gap_nm=cgf_rht["gap_nm"],
                unit_width_nm=cgf_rht["layer_width_nm"],
            )
        states_htc_series[state] = htc

        # get max_htc
        if max_htc < htc:
            max_htc = htc
            logging.info(
                f"Better state found at state number {state_idx}: {state} ({htc})"
            )

        # store plot points
        htc_curve.update(max_htc, state_idx)

    # get states corresponding to max htc
    optimal_states_list = states_htc_series[
        states_htc_series == states_htc_series.max()
    ].index.tolist()  # [3]

    # print execution time of this problem
    timer(start_time, time.time())

    # ------------- output relevant data -------------
    print(
        "Optimal states are:",
        optimal_states_list,
        "\nNumber of equally optimal states:",
        len(optimal_states_list),
    )

    # store htc computed values (if selected)
    if save_computed_htc and state_idx % saving_frequency == 0:
        store_series_to_hdf(
            f"{cgf_rht['htc_path']}/HTC_{cgf_rht['n_layers']}_layers.h5",
            htc_values,
            data_key="value_series",
        )
        logging.info(f"HTC file saved at stage {state_idx}, counting from 1.")

    # plot
    if plot_learning_curves:
        htc_curve.plot(
            title="",
            xlabel="Found states",
            ylabel="Largest HTC ($\mathdefault{W/m^2K}$)",
            plot_epsilon=False,
            plot_lr=False,
            save_path=f"./img/{cfg_dummy['generic_save_folder']}/Dummy_learning_curve_{cgf_rht['n_layers']}.png",
        )

    return htc_curve, states_htc_series
