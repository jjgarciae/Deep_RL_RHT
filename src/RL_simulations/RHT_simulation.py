import copy
import logging
import sys
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from global_vars import WORKING_DIR

# for HPC, specify the path to import from our modules
# .. [1] https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
sys.path.insert(1, WORKING_DIR)  # [1]

# import the rest of all necessary libraries
from src.RL.DRL import DQN_agent, DRL_agent  # noqa E402
from src.RL.environment import RHT  # noqa E402
from src.utils import exists, load_conf, set_logging, store_series_to_hdf, timer


def approximated_simulation(
    algorithm: Literal["Sarsa", "Q-learning", "double_Q-learning"],
    cfg_path: Path = Path("./src/config.toml"),
    save_computed_htc: bool = False,
    save_htc_csv: bool = True,
    save_q_net: bool = True,
    plot: bool = False,
    logging_level: str = "warn",
    seed: Optional[int] = None,
):
    """Make a RL approximated method simulation of the agent in the desired
    environment. We employ neural networks to perform this approximation.

    Parameters
    ----------
    algorithm : Literal["Sarsa", "Q-learning", "double_Q-learning"]
        Determine the RL algorithm to use. Integrated algorithms are:
            * Sarsa
            * Q-learning
            * double Q-learning
    cfg_path : Path, optional
        Configuration file path. By default, Path("./src/config.toml").
    save_computed_htc : bool, optional
        Overwrite stored htc values in order to add new computed values. If
        several runs are executed at the same time, this could lead to execution
        failure.
        By default, False.
    save_htc_csv : bool, optional
        Select to store a csv with the htc value of each state. Useful to
        visualize those values. By default, True.
    save_q_net : bool, optinal
        Select to save the action-state values network.
        By default, True.
    plot : bool, optional
        Select if we desire to obtain different plots, such as:
         - Loss curves
         - Reward curves
        By default, False.
    logging_level : str, optional
        Select logging level. Additionally, if `debug`/`info` are selected, set
        verbose to True, else, to False.
        By default, `warn`.
    seed : Optional[int], optional
        Seed number along the simulation. If None, do not fix any seed.
        By default, None.

    Returns
    -------
    agent : agent
        Trained agent.
    env : environment
        Environment where the agent was trained.
    last_state : str
        Predicted optimal state.
        We will assume optimal state is the state where the agent tends to end.
        Because of that, we will perform a very long simulation with a greedy
        policy and output the end state as the OPTIMAL STATE.

    Warnings
    --------
    `HTC_{n_layers}_layers.h5` input can be problematic.
        If it has been previously **UNsuccessfully** generated, a HDF5ExtError
        will raise.
        To solve it, delete `HTC_{n_layers}_layers.h5` and execute twice
        the simulation, as the first modification to the file will raise a
        ValueError as file is already opened and it is required to be in
        read-only mode.
        Another solution is to delete the problematic file and copy-paste it
        from HTC_backup.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/38537905/set-logging-levels
    .. [2] https://stackoverflow.com/questions/65728037/matplotlib-debug-turn-off-when-python-debug-is-on-to-debug-rest-of-program
    .. [3] https://docs.python.org/3/library/copy.html
    .. [4] https://docs.python.org/3/library/logging.html#levels
    """
    # set logging level [1], [4]
    set_logging(level=logging_level)

    if logging_level in ["debug", "info"]:
        verbose = True
    else:
        verbose = False

    # load the configuration
    cfg_logging = load_conf(cfg_path, key="logging")

    cfg = load_conf(cfg_path, key="RHT")
    cfg_plots = cfg["plots"]

    cfg = load_conf(cfg_path, key="DRL")
    cfg_algorithm = cfg[algorithm]
    cfg_hiperpar = cfg_algorithm["RHT"]["hiperparams"]
    cfg_save_folder = cfg_algorithm["RHT"]["generic_save_folder"]

    # We want to be able to train our model on a hardware accelerator like the
    # GPU or MPS, if available.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logging.info(f"Using {device} device")

    # store start time of the program
    start_time = time.time()

    # load HTC input constants
    # frequencies range
    with open("./data/HTC/HTC_frequencies.npy", "rb") as f:
        freq_range = np.load(f)
    # materials permitivities at each frequencies
    with open("./data/HTC/HTC_permitivities.npy", "rb") as f:
        mat_per = np.load(f)

    # create the RHT environment
    env = RHT(mat_per, freq_range, cfg_path=cfg_path, seed=seed)

    # create the new agent with selected algorithm
    if algorithm == "double_Q-learning":
        agent_class = DQN_agent
    else:
        agent_class = DRL_agent

    agent = agent_class(
        algorithm=algorithm,
        actions=env.action_space,
        save_folder=cfg_save_folder,
        verbose=verbose,
        hidden_layers=cfg_hiperpar["hidden_layers"],
        hidden_neur=cfg_hiperpar["hidden_neur"],
    )

    agent.train(
        device=device,
        environment=env,
        discount_rate=cfg_hiperpar["discount_rate"],
        lr=cfg_hiperpar["lr"],
        reduce_perc_lr=cfg_hiperpar["reduce_perc_lr"],
        min_lr=cfg_algorithm["min_lr"],
        epsilon=cfg_hiperpar["epsilon"],
        reduce_eps=cfg_hiperpar["reduce_eps"],
        min_eps=cfg_algorithm["min_eps"],
        batch_size=cfg_hiperpar["batch_size"],
        max_steps=cfg_hiperpar["max_steps"],
        tol_loss=cfg_hiperpar["tol_loss"],
        plot_learning_curves=plot,
        save_q_net=save_q_net,
        memory_size=cfg_hiperpar.get("memory_size", None),  # only DQN
        n_batch_per_step=cfg_hiperpar.get("n_batch_per_step", None),  # only DQN
        n_new_experiences_per_step=cfg_hiperpar.get(
            "n_new_experiences_per_step", None
        ),  # only DQN
        target_estimation_mode=cfg_hiperpar.get(
            "target_estimation_mode", None
        ),  # only DQN
        n_steps_for_target_net_update=cfg_hiperpar.get(
            "n_steps_for_target_net_update", None
        ),  # only DQN
        episode_start_state=cfg_hiperpar["episode_start_state"],
        decorrelated=cfg_hiperpar["decorrelated"],
        step_count=cfg_hiperpar["step_count"],
        reward_curve_mode=cfg_plots["reward_curve_mode"],
        reward_curve_steps_per_point=cfg_plots["reward_curve_steps_per_point"],
        debug_counter=cfg_logging["debug_counter"],
        seed=seed,
    )

    # print execution time of this problem
    timer(start_time, time.time())

    # ------------- output relevant data -------------
    # Output the OPTIMAL STATE.
    # With this aproximated method, we will assume optimal state is the state
    # where the agent tends to end.
    # Because of that, we will perform a very long simulation with a greedy
    # policy and output the end state as the OPTIMAL STATE.
    _, last_reward, _, last_state, last_action = agent.greedy_simulation(
        q_net=agent.q_net,
        environment=copy.deepcopy(env),  # [3]
        start_state=cfg_hiperpar["episode_start_state"],
        steps=env.n_layers * 10,
        device=device,
    )
    print(
        f"The optimal state is: {last_state}",
        f"\nWith an associated reward of: {'{:.3f}'.format(last_reward)} W/m²K",
    )
    logging.info(f"Associated last action was {last_action}")
    logging.info(
        (
            f"Number of visited states: {len(agent.visited_states)}"
            f"/{len(env.state_space)}"
            if exists(env.state_space)
            else ""
        )
    )
    logging.info(
        (
            f"Number of reached states: {len(agent.max_htc_vs_known_state.iterations)}"
            f"/{len(env.state_space)}"
            if exists(env.state_space)
            else ""
        )
    )
    logging.debug(f"Which are: {agent.visited_states}")

    # store the obtained states htc into hdf and csv (if selected)
    if save_computed_htc:
        store_series_to_hdf(
            f"{env.htc_path}/HTC_{env.n_layers}_layers.h5", env.htc_values
        )

    if save_htc_csv:
        env.htc_values.to_csv(
            f"./data/{cfg_save_folder}/state_{env.n_layers}_layers_htc.csv"
        )

    return agent, env, last_state
