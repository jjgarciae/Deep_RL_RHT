import logging
import sys
import time
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from global_vars import WORKING_DIR

# for HPC, specify the path to import from our modules
# .. [1] https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
sys.path.insert(1, WORKING_DIR)  # [1]

# import the rest of all necessary libraries
from RHT_dummy_simulation import htc_dummy_simulation  # noqa E402
from RHT_simulation import approximated_simulation  # noqa E402

from src.RHT.HTC_calculator import HTC_material_defs
from src.RL.plots import mean_learning_curve, merge_learning_curves_figs
from src.RL.utils import HTC_units_label
from src.utils import (
    InputError,
    exists,
    load_conf,
    load_hdf_to_series,
    set_logging,
    str_to_tuple_or_list,
    timer,
)


def approximated_simulation_runs(
    n_runs: int,
    algorithm: Literal["Sarsa", "Q-learning", "double_Q-learning"],
    cfg_path: Path = Path("./src/config.toml"),
    save_computed_htc: bool = False,
    save_htc_csv: bool = True,
    save_q_nets: bool = True,
    plot: bool = False,
    perc_runs_top_HTC: bool = False,
    logging_level: str = "warn",
    seeds: Optional[list[int]] = None,
) -> mean_learning_curve:
    """Create several Radiative Heat Transfer (RHT) RL approximated method
    simulations and obtain their promediated results, such as:
     - Number of visisted state-actions and states
     - Learning curves
     - etc.

    Parameters
    ----------
    n_runs : int
        Number of runs to promediate.
    algorithm : Literal["Sarsa", "Q-learning", "double_Q-learning"]
        Algorithm to obtain the "optimal state" from RHT simulation.
    cfg_path : Path, optional
        General path where all the configuration relies.
        By default, Path("./src/config.toml").
    save_computed_htc : bool, optional
        Each run, overwrite stored htc values in order to add new computed
        values. If several runs are executed at the same time, this could lead
        to execution failure.
        By default, False.
    save_htc_csv : bool, optional
        Select if we want to save htc value of each state into a cvs.
        By default, True.
    save_q_nets : bool, optional
        Select if we want to save the computed q_net of each run.
        By default, True.
    plot : bool, optional
        Select if we desire to obtain different plots, such as:
         - Loss curves
         - Reward curves
        By default, False.
    perc_runs_top_HTC : bool, optional
        Show the percentage of runs which reach the maximum and the top 5 HTC of
        those HTC discovered during this simulation.
        By default, False.
        * This option is only recommended when all HTC values have been
        previously stored, otherwise this output can be misleading as we won´t
        know the true best HTC.
    logging_level : str, optional
        Select logging level.
        By default, `warn`.
    seeds : Optional[list[int]], optional
        Select each one of the seeds to apply to each one of the runs. If None,
        do not fix any seed.
        By default, None.

    Return
    ------
    mean_max_htc_known_state : mean_learning_curve
        Store unique known states and the corresponding max htc discovered each
        time a new state is known.

    See Also
    ----------
    approximated_simulation : function that performs each run.

    Warnings
    --------
    It is assumed that all runs are equally defined.
    The only difference among them must be stochastic phenomena (such as random
    walk, random network initialization, etc.).

    References
    ----------
    .. [1] https://stackoverflow.com/questions/14324477/bold-font-weight-for-latex-axes-label-in-matplotlib
    .. [2] https://stackoverflow.com/questions/72792392/how-to-use-latex-within-an-f-string-expression-in-matplotlib-no-variables-in-eq
    .. [3] https://stackoverflow.com/questions/35871907/pandas-idxmax-best-n-results
    """
    # input checks
    if exists(seeds) and len(seeds) != n_runs:
        raise InputError("Number of selected seeds and runs must be the same.")
    elif seeds is None:
        seeds = np.repeat(None, n_runs)

    # set logging level
    set_logging(level=logging_level)

    # load the configuration
    cfg = load_conf(cfg_path, key="RHT")
    n_layers = cfg["environment"]["n_layers"]
    n_materials = len(cfg["environment"]["materials"])

    cfg_plots = cfg["plots"]
    reward_curve_mode = cfg_plots["reward_curve_mode"]
    reward_curve_steps_per_point = cfg_plots["reward_curve_steps_per_point"]

    cfg = load_conf(cfg_path, key="DRL")
    cfg_save_folder = cfg[algorithm]["RHT"]["generic_save_folder"]

    # Output data is stored as an attribute in each agent object. We must
    # promediate the result of each one of the agents.
    # Get the agent, environment of each simulation and promediate all ouptus.
    agent_list = []
    env_list = []
    mean_last_state = np.zeros(n_layers)

    # store start time of the program
    RUNS_start_time = time.time()

    for run in range(n_runs):
        logging.info(f"\n\nComputing run number {run}...\n\n")

        agent, env, last_state = approximated_simulation(
            algorithm=algorithm,
            cfg_path=cfg_path,
            save_computed_htc=save_computed_htc,
            save_htc_csv=False,
            plot=False,
            save_q_net=False,
            logging_level=logging_level,
            seed=seeds[run],
        )
        agent_list.append(agent)
        env_list.append(env)
        mean_last_state += (
            np.array(str_to_tuple_or_list(last_state, to="list")) / n_runs
        )

    # print execution time for all simulations
    timer(RUNS_start_time, time.time())

    # average number of visited actions / states
    n_mean_visited_states = (
        sum([len(agent.visited_states) for agent in agent_list]) / n_runs
    )

    print(
        f"\nMean number of visited states: {n_mean_visited_states}",
        f"/{len(env_list[0].state_space)}" if exists(env_list[0].state_space) else "",
    )

    # compute `mean_max_htc_known_state` even if plot input is false as it is a
    # function output
    mean_max_htc_known_state = mean_learning_curve(
        [agent.max_htc_vs_known_state for agent in agent_list], xrange="min"
    )

    # output percentaje of runs with max/top 5 HTC values explored
    if perc_runs_top_HTC:
        # retrieve best HTC and top 5 best HTC
        best_HTC = env.htc_values.max()
        top5_HTC = env.htc_values.nlargest(5).values.tolist()  # [3]

        # check percentage of runs in which the best/top 5 HTC is discovered
        # with exploration for:
        # - all the complete simulations
        # - until `mean_max_htc_known_state` plot represented range
        #   (CAUTION: this can be done as `mean_learning_curve` xrange is `min`)
        crop_to_plot_range = True
        plot_range = mean_max_htc_known_state.iterations.shape[0]
        n_perc_outputs = 2  # two percentage checks
        for _ in range(n_perc_outputs):

            if crop_to_plot_range:
                # crop explored HTCs to plot_range
                range_slice = slice(plot_range)
            else:
                # get all explored HTCs during each simulation
                range_slice = slice(None)

            # ------------------ best ----------------
            n_runs_best_state_reached = sum(
                [
                    best_HTC in agent.max_htc_vs_known_state.performance[range_slice]
                    for agent in agent_list
                ]
            )
            perc_best = n_runs_best_state_reached / n_runs * 100

            print(
                f"\n{perc_best}% of runs explored best HTC, with a value of {best_HTC} W/m²K. "
            )

            # ------------------ top 5 ----------------
            n_runs_top5_state_reached = sum(
                [
                    len(
                        set(
                            agent.max_htc_vs_known_state.performance[range_slice]
                        ).intersection(set(top5_HTC))
                    )
                    > 0
                    for agent in agent_list
                ]
            )
            perc_top_5 = n_runs_top5_state_reached / n_runs * 100

            print(f"{perc_top_5}% of runs explored top 5 best HTC. ")

            # once the percentajes of the cropped case are calculated, turn
            # crop_to_plot_range falg to false
            if crop_to_plot_range:
                print(f"Taken limit of {plot_range} explored states. ")
            crop_to_plot_range = False

        print(
            f"\nMax HTC values considered above with {len(env.htc_values)} of {n_materials**n_layers} states.\n"
        )

    # load state rewards and store them in a csv
    if save_htc_csv:
        env.htc_values.to_csv(
            f"./data/several_RUNS/{cfg_save_folder}/HTC_{n_layers}_layers.csv"
        )

    # save trained q networks
    if save_q_nets:
        for idx, agent in enumerate(agent_list):
            torch.save(
                agent.q_net.state_dict(),
                f"./data/several_RUNS/{cfg_save_folder}/q_net_{n_layers}_inputs_run{idx}.pt",
            )

    # plots
    if plot:
        # obtain units label
        units_label = HTC_units_label(env.reward_reduction_factor)

        # Generate the mean of learning curves and plot them
        mean_loss_curve = mean_learning_curve(
            [agent.loss_curve for agent in agent_list], xrange="min"
        )

        mean_loss_curve.plot(
            title="",
            xlabel="Training step",
            ylabel="MAE loss",
            plot_epsilon=cfg_plots["epsilon"],
            plot_lr=cfg_plots["lr"],
            save_path=f"./img/several_RUNS/{cfg_save_folder}/Loss_learning_curve_{n_layers}.png",
        )

        mean_max_htc_known_state.plot(
            title="",
            xlabel="Found states",
            ylabel=f"Largest HTC ({units_label})",
            y_divisor=env.reward_reduction_factor,
            plot_epsilon=False,
            plot_lr=False,
            save_path=f"./img/several_RUNS/{cfg_save_folder}/Max_htc_per_known_state_{n_layers}.png",
        )

        if exists(reward_curve_steps_per_point):
            for reward_curve_mode_ in reward_curve_mode:
                if reward_curve_mode_ == "return":
                    mean_reward_curve = mean_learning_curve(
                        [agent.reward_curve_return for agent in agent_list],
                        xrange="min",
                    )
                    reward_ylabel = f"Return ({units_label})"
                    sufix = "return"

                elif reward_curve_mode_ == "last_reward":
                    mean_reward_curve = mean_learning_curve(
                        [agent.reward_curve_last_reward for agent in agent_list],
                        xrange="min",
                    )
                    reward_ylabel = f"Last reward of greedy simulation ({units_label})"
                    sufix = "last_reward"

                elif reward_curve_mode_ == "best_reward":
                    mean_reward_curve = mean_learning_curve(
                        [agent.reward_curve_best_reward for agent in agent_list],
                        xrange="min",
                    )
                    reward_ylabel = f"Maximum reward ({units_label})"
                    sufix = "best_reward"

                elif reward_curve_mode_ == "best_htc":
                    mean_reward_curve = mean_learning_curve(
                        [agent.reward_curve_best_htc for agent in agent_list],
                        xrange="min",
                    )
                    reward_ylabel = f"Maximum heat transfer coefficient ({units_label})"
                    sufix = "best_htc"

                mean_reward_curve.plot(
                    title="",
                    xlabel="Training step",
                    ylabel=reward_ylabel,
                    plot_epsilon=cfg_plots["epsilon"],
                    plot_lr=cfg_plots["lr"],
                    y_divisor=(
                        env.reward_reduction_factor
                        if reward_curve_mode_ == "best_htc"
                        else None
                    ),
                    save_path=f"./img/several_RUNS/{cfg_save_folder}/Reward_learning_curve_{n_layers}_{sufix}.png",
                )
    return mean_max_htc_known_state


def dummy_simulation_runs(
    n_runs: int,
    cfg_path: Path = Path("./src/config.toml"),
    save_computed_htc: bool = False,
    plot: bool = False,
    perc_runs_top_HTC: bool = False,
    logging_level: str = "warn",
    seeds: Optional[list[int]] = None,
) -> mean_learning_curve:
    """Obtain the "optimal state" obtained with Reinforcement Learning for the
    Radiative Heat Transfer simulation naively, this is, compute the
    Heat-Transfer Coefficient (HTC) of all states and select the state wich
    returns the highest HTC.

    Parameters
    ----------
    n_runs : int
        Number of runs to promediate.
    cfg_path : Path, optional
        Config path.
        By default, Path("./src/config.toml")
    save_computed_htc : bool, optional
        Overwrite stored htc values in order to add new computed values. If
        several runs are executed at the same time, this could lead to execution
        failure.
        By default, False.
    plot : bool, optional
        Select to plot the mean of `best htc` curves or not.
        By defaul, False
    perc_runs_top_HTC : bool, optional
        Show the percentage of runs which reach the maximum and the top 5 HTC of
        those HTC discovered during this simulation.
        By default, False.
        * This option is only recommended when all HTC values have been
        previously stored, otherwise this output can be misleading as we won´t
        know the true best HTC.
    logging_level : str, optional
        Select logging level.
        By default, `warn`.
    seeds : Optional[list[int]], optional
        Select each one of the seeds to apply to each one of the runs. If None,
        do not fix any seed.
        By default, None.

    Returns
    -------
    htc_curve_MEAN : mean_learning_curve
        Object with the mean of `best htc` curves.

    Warnings
    --------
    `HTC_{n_layers}_layers.h5` input can be problematic.
        If it has been previously **UNsuccessfully** generated, a HDF5ExtError
        will raise.
        To solve it, delete `HTC_{n_layers}_layers.h5` and execute twice
        the simulation, as the first modification to the file will raise a
        ValueError as file is already opened and it is required to be in
        read-only mode.
    """
    # input checks
    if exists(seeds) and len(seeds) != n_runs:
        raise InputError("Number of selected seeds and runs must be the same.")
    elif seeds is None:
        seeds = np.repeat(None, n_runs)

    # set logging level
    set_logging(level=logging_level)

    # load the configuration
    cgf_rht = load_conf(cfg_path, key="RHT")["environment"]
    cfg_dummy = load_conf(cfg_path, key="dummy")

    # store start time of the program
    RUNS_start_time = time.time()

    ALL_htc_curves = []
    ALL_states_htc_series = []
    for run in range(n_runs):
        logging.info(f"\n\nComputing run number {run}...\n\n")

        htc_curve, states_htc_series = htc_dummy_simulation(
            subset_size=cfg_dummy["subset_size"],
            cfg_path=cfg_path,
            save_computed_htc=save_computed_htc,
            plot_learning_curves=False,
            logging_level=logging_level,
            seed=seeds[run],
        )
        ALL_htc_curves.append(htc_curve)
        ALL_states_htc_series.append(states_htc_series)

    htc_curve_MEAN = mean_learning_curve(ALL_htc_curves, xrange="min")

    # print execution time for all simulations
    timer(RUNS_start_time, time.time())

    # Get all relevant data
    # output percentaje of runs with max/top 5 HTC values explored
    if perc_runs_top_HTC:
        # DATA
        # obtain all available `htc_values`
        htc_values = load_hdf_to_series(
            f"{cgf_rht['htc_path']}/HTC_{cgf_rht['n_layers']}_layers.h5"
        )
        # obtain number of materials and layers to compute total number of
        # states
        n_materials = len(cgf_rht["materials"])
        n_layers = cgf_rht["n_layers"]

        # PERCENTAGE
        # retrieve best HTC and its state, and top 5 best HTC states
        best_HTC = htc_values.max()
        best_HTC_state = htc_values.idxmax()
        top5_HTC_states = htc_values.nlargest(5).index.tolist()

        # check percentage of runs in which the best HTC is discovered with
        # exploration
        n_runs_best_state_reached = sum(
            [
                best_HTC_state in states_htc_series.index.tolist()
                for states_htc_series in ALL_states_htc_series
            ]
        )
        perc_best = n_runs_best_state_reached / n_runs * 100
        print(
            f"\n{perc_best}% of runs explored best HTC, with a value of {best_HTC} W/m²K. "
        )

        # check percentage of runs in which at least one of the top 5 HTC is visited
        n_runs_top5_state_reached = sum(
            [
                len(set(states_htc_series.index).intersection(set(top5_HTC_states))) > 0
                for states_htc_series in ALL_states_htc_series
            ]
        )
        perc_top5 = n_runs_top5_state_reached / n_runs * 100
        print(
            f"{perc_top5}% of runs explored top 5 best HTC. "
            f"\nBest values provided with {len(htc_values)} of {n_materials**n_layers} states.\n"
        )

    # plots
    if plot:
        htc_curve_MEAN.plot(
            title="",
            xlabel="Found states",
            ylabel="Largest HTC ($\mathdefault{W/m^2K}$)",
            plot_epsilon=False,
            plot_lr=False,
            save_path=f"./img/several_RUNS/{cfg_dummy['generic_save_folder']}/Dummy_learning_curve_{cgf_rht['n_layers']}.png",
        )

    return htc_curve_MEAN


def approximated_nd_dummy_simulation_runs(
    n_runs: int,
    apprx_algorithm: Literal["Sarsa", "Q-learning", "double_Q-learning"],
    cfg_path: Path = Path("./src/config.toml"),
    save_computed_htc: bool = False,
    save_htc_csv: bool = True,
    apprx_save_q_nets: bool = True,
    plot: bool = False,
    perc_runs_top_HTC: bool = False,
    logging_level: str = "warn",
    seeds: Optional[list[int]] = None,
):
    """Generate approximated and dummy simulations in order to compare them in a shared plot.

    Warnings
    --------
    If selected, this function can save products in the following folders:
    * data/HTC
    * data/several_RUNS/DRL/RHT_{algorithm}
    * img/several_RUNS/DRL/RHT_{algorithm}
    * img/several_RUNS/dummy
    * img/several_RUNS/apprx_nd_dummy/RHT_{algorithm}
    Take this into account in order to avoid any FileNotFound error.

    See Also
    --------
    Unexplained input parameters are explained in their corresponding fucntion.
    Those parameters with `apprx` suffix are only employed for
    `approximated_simulation_runs`.
    * approximated_simulation_runs : approximated simulation code.
    * dummy_simulation_runs : dummy simulation code.
    """
    cfg_env = load_conf(cfg_path, key="RHT")["environment"]
    cfg_apprx_nd_dummy = load_conf(cfg_path, key="apprx_nd_dummy")

    cfg = load_conf(cfg_path, key="DRL")
    cfg_save_folder = cfg[algorithm]["RHT"]["generic_save_folder"]

    # execute the APPROXIMATED simulation
    print(f"\n Computing {apprx_algorithm} algorithm...\n")
    mean_max_htc_known_state = approximated_simulation_runs(
        n_runs=n_runs,
        algorithm=apprx_algorithm,
        cfg_path=cfg_path,
        save_computed_htc=save_computed_htc,
        save_htc_csv=save_htc_csv,
        save_q_nets=apprx_save_q_nets,
        plot=plot,
        perc_runs_top_HTC=perc_runs_top_HTC,
        logging_level=logging_level,
        seeds=seeds,
    )

    # execute the dummy simulation
    print("\n Computing dummy algorithm...\n")
    htc_curve_MEAN = dummy_simulation_runs(
        n_runs=n_runs,
        cfg_path=cfg_path,
        save_computed_htc=save_computed_htc,
        plot=plot,
        perc_runs_top_HTC=perc_runs_top_HTC,
        logging_level=logging_level,
        seeds=seeds,
    )

    apprx_nd_dummy = merge_learning_curves_figs(
        [mean_max_htc_known_state, htc_curve_MEAN], curve_fill=None
    )

    # obtain units label
    units_label = HTC_units_label(cfg_env["reward_reduction_factor"])
    apprx_nd_dummy.plot(
        title="",
        xlabel="Found states",
        ylabel=f"Largest HTC ({units_label})",
        y_divisor=cfg_env["reward_reduction_factor"],
        save_path=f"./img/several_RUNS/{cfg_apprx_nd_dummy['generic_save_folder']}/{cfg_save_folder.split('/')[1]}/Max_htc_per_seen_state_{cfg_env['n_layers']}.png",
        labels=["RL", "Random"],
    )


if __name__ == "__main__":
    # Compute HTC input constants. They will be loaded into the simulation from
    # the saved route.
    HTC_material_defs(0.3, 3, 200, save_path="data/HTC")

    # set config route and algorithm to execute
    n_runs = 10
    config_path = Path("./src/config.toml")
    algorithm = "Sarsa"

    # execute the dummy and APPROXIMATED simulation
    approximated_nd_dummy_simulation_runs(
        n_runs=n_runs,
        apprx_algorithm=algorithm,
        cfg_path=config_path,
        save_computed_htc=False,
        save_htc_csv=True,
        apprx_save_q_nets=True,
        plot=True,
        perc_runs_top_HTC=True,
        logging_level="warn",
        seeds=None,
    )
    plt.close(fig="all")
