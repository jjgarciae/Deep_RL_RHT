import itertools
import random
import warnings
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

from src.RHT.HTC_calculator import HTC
from src.RL.basics import environment
from src.utils import (
    InputError,
    exists,
    load_hdf_to_series,
    load_or_update_series,
    str_to_tuple_or_list,
    tuple_or_list_to_str,
)


class RHT(environment):
    """Environment of Radiative Heat Transfer between two bodies.
    It is characterized by:
        * Distance between cristals.
        * The multilayer structures between them.
    Here, we consider structures composed by two materials: alternating layers
    of dielectric and metal.

    Notes
    -----
    * N layers of the same material can be repeated, so they will be considered
        as a single layer of width `N*single_layer_width`.
    * The considered metal is made up, is not real. It is similar to the
        behaviour to the SiC SPhP. It behaves as a metal on the IR range.

    References
    ----------
    .. [1] https://gymnasium.farama.org/
    """

    def __init__(
        self,
        mat_per: np.ndarray,
        freq_range: np.ndarray,
        cfg_path: Path = Path("./src/config.toml"),
        htc_path: Optional[str] = "./data/HTC",
        n_layers: int = 3,
        start_state: Union[Literal["rand", "zeros"], str] = "zeros",
        gap_nm: int = 10,
        layer_width_nm: int = 5,
        n_episode_steps: int = 6,
        reward_def: Literal["htc", "htc_delta"] = "htc",
        ref_state_label: Optional[str] = None,
        reward_reduction_factor: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """Initialization of the RHT environment. It is componsed by multilayered
        structures with a vacuum gap between the two bodies of study.

        Parameters
        ----------
        mat_per : np.ndarray
            Permitivities of each material for the frequencies of study.
        freq_range: np.ndarray
            Frequencies of study.
        cfg_path : Path, optional
            Config path.
            By default, Path("./src/config.toml")
        htc_path : Optional[str], optional
            Path where htc values for each state are stored.
            If None, just compute all necessary htc values and do not store them.
            By default, "./data/HTC"
        n_layers : int, optional
            Number of layers which composes the multilayer structure.
            By default, 3.
        start_state : Union[Literal["rand", "zeros"], str], optional
            Start state of the environment.
            * Select "rand" to initialize randomly.
            * Select "zeros" to initialize all layers to 0.
            By default, "zeros".
        gap_nm : int, optional
            Gap between the two bodies, in nm.
            By default, 10.
        layer_width_nm : int, optional
            Widths of each layer, in nm.
            By default, 5.
        n_episode_steps : int, optional
            Number of steps to take on each episode.
            By default, 6.
        reward_def : Literal["htc", "htc_delta"], optional
            Reward of the environment. Two options:
                * htc : The reward is the htc value of reached state.
                * htc_delta : The reward is the difference between htc values.
            By default, "htc".
        ref_state_label: Optional[str], optional
            Determines the reward when "htc_delta" is selected.
                reward = next_state_htc(s') - ref_state_htc
            If `ref_state_label` is:
                * None: take s state of (s, a, r, s') as reference state.
            By default, None.
        reward_reduction_factor : Optional[float], optional
            Select the reduction factor of the reward.
            Useful if the reward is too high, so a bad convergence of neural
            networks could be arised.
            By default, None.
        seed : Optional[int], optional
            Random seed to get always the same initial state.
            If None, do not apply any seed.
            By default, None.

        See Also
        --------
        * HTC_material_defs : function which computes `mat_per` and
          `freq_range`.
        * `load_or_update_series` / `load_hdf_to_series`: useful functions to
          retrieve htc values.

        Notes
        -----
        The majority of optional values will be taken from specified config path.
        Else, they will be taken from function input.

        References
        ----------
        .. [1] https://stackoverflow.com/questions/104420/how-do-i-generate-all-permutations-of-a-list
        .. [2] https://stackoverflow.com/questions/50992063/set-a-tuple-as-column-name-in-pandas
        """
        # inherit from parent class
        super().__init__(cfg_path=cfg_path)

        # fix seed if indicated
        if exists(seed):
            random.seed(seed)

        # get environment configuration
        self.cfg = self.cfg["RHT"]["environment"]

        # define some static attributes
        self.materials = self.cfg["materials"]
        self.mat_per = mat_per
        self.freq_range = freq_range

        self.n_layers = self.cfg.get("n_layers", n_layers)
        self.gap = self.cfg.get("gap_nm", gap_nm)
        self.layer_width = self.cfg.get("layer_width_nm", layer_width_nm)

        self.n_episode_steps = self.cfg.get("n_episode_steps", n_episode_steps)

        # set the reward
        self.reward_def = self.cfg.get("reward_def", reward_def)
        self.ref_state_label = self.cfg.get("ref_state_label", ref_state_label)
        self.reward_reduction_factor = self.cfg.get(
            "reward_reduction_factor", reward_reduction_factor
        )

        # initialize episode count
        self.episode_count = 0

        # define some attributes analogous to gym [1], [2]
        try:
            tuple_state_space = list(
                itertools.product(self.materials, repeat=self.n_layers)
            )
            self.state_space = tuple_or_list_to_str(tuple_state_space)
        except MemoryError:
            warnings.warn(
                "`self.state_space` could not be generated due to memory shortage."
            )
            self.state_space = None

        tuple_action_space = list(
            itertools.product(self.materials, list(range(self.n_layers)))
        )
        self.action_space = tuple_or_list_to_str(tuple_action_space)

        # initialize the selected material for each layer
        start_state = self.cfg.get("start_state", start_state)
        if start_state == "zeros":
            # set all layers to zero
            start_state = tuple_or_list_to_str(
                [np.zeros(self.n_layers, dtype=int).tolist()]
            )[0]
        elif start_state == "rand":
            # get a str with a random state
            start_state = self.random_state_label_choice()
        self.start_state = start_state

        # get stored htc values or initialize its storage into RAM
        self.htc_path = self.cfg.get("htc_path", htc_path)
        if exists(self.state_space):
            self.htc_values = load_hdf_to_series(
                f"{self.htc_path}/HTC_{self.n_layers}_layers.h5",
                all_index=self.state_space,
            )
        else:
            self.htc_values = pd.Series()

    def reset(
        self,
        init_layers: Union[Literal["initial", "rand"], str] = "initial",
        seed: Optional[int] = None,
    ) -> str:
        """Reset the environment to its basal state:
            * Return initial state
            * Episode counter is reset to zero

        Parameters
        ----------
        init_layers :  ["initial", "rand", str], optional
            Initialization of the materials of the multilayer structure. Here,
            we only consider two materials:
            * Dielectric : represented by a 0. Vacuum.
            * Metal : represented by a 1. It is a metal we have made up, it is
            not real!
            Select "initial" to use the environment initial state.
            By default, "inital".
        seed : Optional[int], optional
            Random seed to get always the same initial state.
            If None, do not apply any seed.
            By default, None.

        Returns
        -------
        str
            Initial state.

        Examples
        --------
        >>> env.reset(init_layers = "0, 0, 1")
        >>> env.reset(init_layers = "initial")
        >>> env.reset(init_layers = "rand", seed = 33)
        """
        # check if input state if given as a string
        if not isinstance(init_layers, str):
            raise InputError("Input layers must be given as string. Ex. '0, 0, 1'")

        # fix seed if indicated
        if exists(seed):
            random.seed(seed)

        # random initial state
        if init_layers == "rand":
            init_layers = self.random_state_label_choice()

        # initial state from environment initial state
        elif init_layers == "initial":
            init_layers = self.start_state

        # reset episode conuter to zero
        self.episode_count = 0

        return init_layers

    def step(
        self, state_label: str, action_label: str, **kwargs
    ) -> tuple[str, float, bool]:
        """Returns the next state and reward given an input state and action.
        Aditionally, output if the episode is finished, defined by if the
        early_stopping counter has reached the specified value by `patience`.

        Parameters
        ----------
        state_label : str
            Current state label of the agent.
        action_label : str
            Action label taken by the agent.
            First value is selected material, second value is its position.

        ** kwargs
            step_count : bool, optional
                Select to have an early_stopping counter.
                Specially usefull for those algorithms that calls this `step`
                method more than once per training step.
                By default, False, so early_stopping is inactive.

        Returns
        -------
        next_state_label: str
            Next state label of the agent
        reward : float
            Reward due to the action from input state.
        end_episode : bool
            Aditional output if we have an episodic task.
            If True, an episode has ended.

        Notes
        -----
        `self.htc_values` are being updated if missing values are found. If you
        want to preserve these values, remember to store it at the end of the
        simulation.
        Storage along the simulation is not recommended due performance reasons.
        """
        step_count = kwargs.get("step_count", False)

        # transform state from str to list to make the pertinent calculous
        state_list = str_to_tuple_or_list(state_label, to="list")
        # transform action to tuple
        action_tuple = str_to_tuple_or_list(action_label, to="tuple")

        # store reference state htc if necessary
        if self.reward_def == "htc_delta":
            # select reference state for reward determination
            if self.ref_state_label is None:
                # reference state will be input state (s)
                ref_state_label = state_label
                ref_state_list = state_list
            else:
                # reference state will be fixed and that specified at `__init__`
                # method
                ref_state_label = self.ref_state_label
                ref_state_list = str_to_tuple_or_list(self.ref_state_label, to="list")

            ref_state_htc, self.htc_values = load_or_update_series(
                self.htc_values,
                desired_index=ref_state_label,
                function=HTC,
                empty_value=np.nan,
                htc_material=ref_state_list,  # HTC function inputs
                mat_per=self.mat_per,
                mat_w=self.freq_range,
                gap_nm=self.gap,
                unit_width_nm=self.layer_width,
            )

        # perform the action over the environment
        state_list[action_tuple[1]] = action_tuple[0]

        # transform next state from list to str
        next_state_label = tuple_or_list_to_str([state_list])[0]

        # retrieve the htc value from a stored file or compute and store it
        if self.htc_path is None:
            next_state_htc = HTC(
                htc_material=state_list,
                mat_per=self.mat_per,
                mat_w=self.freq_range,
                gap_nm=self.gap,
                unit_width_nm=self.layer_width,
            )
        else:
            next_state_htc, self.htc_values = load_or_update_series(
                self.htc_values,
                desired_index=next_state_label,
                function=HTC,
                empty_value=np.nan,
                htc_material=state_list,  # HTC function inputs
                mat_per=self.mat_per,
                mat_w=self.freq_range,
                gap_nm=self.gap,
                unit_width_nm=self.layer_width,
            )

        # return the reward: htc or htc increment
        if self.reward_def == "htc":
            reward = next_state_htc

        elif self.reward_def == "htc_delta":
            reward = next_state_htc - ref_state_htc

        # reduce reward value if selected
        if self.reward_reduction_factor is not None:
            reward = reward / self.reward_reduction_factor

        # add one to the episode count if early stopping active
        if step_count:
            self.episode_count += 1

        # manage the lenght of the episode
        end_episode = False
        if self.episode_count >= self.n_episode_steps:
            self.episode_count = 0
            end_episode = True

        return next_state_label, reward, end_episode

    def random_state_label_choice(self):
        # get a str with a random state
        return (
            random.choice(self.state_space)
            if exists(self.state_space)
            else tuple_or_list_to_str(
                [random.choices(self.materials, k=self.n_layers)]
            )[0]
        )


if __name__ == "__main__":
    # load HTC input constants
    # frequencies range
    with open("./data/HTC/HTC_frequencies.npy", "rb") as f:
        freq_range = np.load(f)
    # materials permitivities at each frequencies
    with open("./data/HTC/HTC_permitivities.npy", "rb") as f:
        mat_per = np.load(f)

    # init HTC environment
    env = RHT(mat_per, freq_range)

    # initialize the environment
    state = env.reset()

    # take a some trial steps in the environment
    for n_trial in range(0, 6):
        action = (np.random.choice(env.materials), random.randint(0, env.n_layers - 1))
        action = tuple_or_list_to_str([action])[0]

        state, reward, end_episode = env.step(state, action)

        print(
            f"Trial {n_trial}: \n\t action {action} \n\t state {state}"
            f"\n\t reward {reward} \n\t end_episode {end_episode}"
        )

        if end_episode:
            state = env.reset()
