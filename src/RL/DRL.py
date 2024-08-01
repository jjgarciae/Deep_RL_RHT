import copy
import logging
import random
import warnings
from collections import deque, namedtuple
from collections.abc import Iterable
from numbers import Number
from typing import Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.RL.basics import agent, environment
from src.RL.plots import learning_curve
from src.RL.utils import HTC_units_label
from src.utils import InputError, exists, str_to_tuple_or_list

transition = namedtuple("transition", ("state", "action_idx", "reward", "next_state"))
sarsa_transition = namedtuple(
    "transition", ("state", "action_idx", "reward", "next_state", "next_action")
)


class ReplayMemory(object):
    """Store experiences collection (s,a,r,s') to train Reinforcement
    Learning algorithm. Tuned code from Pytorch. [1]

    References
    ----------
    .. [1] https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory
    .. [2] https://docs.python.org/3/library/collections.html#collections.deque
    """

    def __init__(self, capacity, agent, **kwargs):
        experiences, self.visited_states, self.reached_states, _, _ = (
            agent._experience_generation(**kwargs)
        )
        self.memory = deque(experiences, maxlen=capacity)

    def push(self, experiences: list[transition]):
        """Save input experiences and visited states
        When `capacity` is reached, `deque` iterator automatically remove older
        elements when new ones are appended. [2]
        """
        for experience in experiences:
            self.memory.append(experience)
            self.visited_states.add(experience.state)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QNN(nn.Module):
    """Neural Network for estimating all the action values from a given state.

    References
    ----------
    .. [1] https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html#define-the-class
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_layers: int, hidden_neur: int):
        """Init function of the action value network.

        Parameters
        ----------
        in_dim : int
            Number of inputs of the action value network.
            It will be equal to the number of the features of a state.
        out_dim : int
            Number of outputs of the action value network.
            It will be equal to the number of actions or q values for a single
            state.
        hidden_layers : int
            Number of desired hidden layers.
        hidden_neur : int
            Number of neurons of each one of the hidden layers.
        """
        super().__init__()

        # add input layer
        layers = [nn.Linear(in_dim, hidden_neur), nn.SELU()]

        # Add hidden layers
        for _ in range(hidden_layers):
            # Append the hidden layer and its corresponding activation function
            layers.append(nn.Linear(hidden_neur, hidden_neur))
            layers.append(nn.SELU())

        # add the output layer
        layers.append(nn.Linear(hidden_neur, out_dim))

        # set the network
        self.selu_stack = nn.Sequential(*layers)

    def forward(self, x):
        y = self.selu_stack(x)
        return y

    def copy_weights(self, net_to_copy):
        """Copy the weights of a give network.

        References
        ----------
        .. [1] Fundations of Deep Reinforcement Learning; Laura Graesser and Wah Loon Keng
        """
        self.load_state_dict(net_to_copy.state_dict())


class DRL_agent(agent):
    """Deep Reinforcement Learning agent.

    References
    ----------
    .. [1] https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(
        self,
        algorithm: Literal["Sarsa", "Q-learning", "double_Q-learning"],
        actions: Iterable[str, Number],
        save_folder: str = "DRL_results",
        verbose: bool = False,
        **kwargs,
    ):
        """Base Deep Reinforcement Learning algorithm for agents.

        Parameters
        ----------
        algorithm : Literal["Sarsa", "Q-learning", "double_Q-learning"]
            Determine the Reinforcement Learning algorithm to use. Implemented
            algorithm are:
                * Sarsa
                * Q-learning
                * double Q-learning
        states : Iterable[str, Number]
            Collection of all possible states of the problem.
        actions : Iterable[str, Number]
            Collection of all possible actions of the problem.
        save_folder : str, optional
            Default name of the save folder for the outputs of the algorithm.
            By default, "DRL_outputs"
        verbose : bool, optional
            Select verbose though the algorithm.
            By default, False

        Other parameters
        ----------------
        **kwargs
            hidden_layers : int, optional
                Number of hidden layers of the action values neural network.
                By default, 2
            hidden_neur : int, optional
                Number of neurons of each one of the hidden layers of the action
                values neural network.
                By default, 16
        """
        # inherit from parent class
        super().__init__(algorithm, actions, verbose)

        # outputs save folder
        self.save_folder = save_folder

        # store more relevant attributes for the algorithm
        self.hidden_layers = kwargs.get("hidden_layers", 2)
        self.hidden_neur = kwargs.get("hidden_neur", 16)

    def greedy_simulation(
        self,
        q_net: QNN,
        environment: environment,
        start_state: str,
        steps: int,
        device: Literal["cuda", "mps", "cpu"],
        step_count: bool = False,
    ) -> tuple[float, float, float, str, str]:
        """Create an small greedy simulation with the current Q network and a
        reset environment.

        Parameters
        ----------
        q_net : QNN
            Network for the prediction of all action-state values for a given
            state.
        environment : environment
            Environment object of the problem.
        start_state : str
            Start state of the simulation.
        steps : int
            Number of steps of the simulation.
        device : Literal["cuda", "mps", "cpu"], optional
            Currently used device for training. Used as an `act` method input.
        step_count : bool, optional
            Select to have an steps counter.
            Specially usefull for those algorithms which do not have a natural
            terminal state, so it is implemented as an episode lenght.
            By default, False, so step count is inactive.

        Returns
        -------
        overall_return : float
            Value of the return for the simulation.
        last_reward : float
            Value of the last reward of the simulation.
        best_reward : float
            Value of the best reward seen durig the simulation.
        last_state : str
            Last state visited. Useful to continue the trajectory of (s, a, r,
            s') generated.
        last_action : str
            Last action performed. For informative purposes.

        Warnings
        --------
        It is recommended that input environment is an independent copy of the
        environment used for the rest algorithm, as it will be reset.

        See Also
        --------
        environment.py : Where environment.reset() method is defined.
        """
        # initialize some parameters
        step = 1
        end_episode = False
        overall_return = 0
        best_reward = -np.inf

        # reset the environment for this simulation and select start state
        state_label = environment.reset(init_layers=start_state)

        # generate the simulation
        while step < steps or end_episode:
            # get action with greedy policy, as we want to evaluate the
            # optimality of the `q_net`
            _, _, action_label = self._act(
                "greedy",
                state_label,
                q_net=q_net,
                device=device,
            )

            # observe response of the environment
            state_label, reward, end_episode = environment.step(
                state_label, action_label, step_count=step_count
            )

            # store best reward of the simulation
            if reward > best_reward:
                best_reward = reward

            # store the return of the simulation
            overall_return += reward

            if end_episode:
                warnings.warn(
                    f"Simulation finalized at step {step} due to end of episode."
                )
                break

            step += 1

        # store last reward value and last state and action labels
        last_reward = reward
        last_state = state_label
        last_action = action_label

        return overall_return, last_reward, best_reward, last_state, last_action

    def _experience_generation(
        self,
        n_experiences: int,
        q_net: QNN,
        environment: environment,
        device: Literal["cuda", "mps", "cpu"],
        epsilon: float,
        initial_state: Union[Literal["rand"], str],
        initial_action: Optional[dict[str, int]] = None,
        follow_next_action: bool = False,
        decorrelated: bool = False,
        **kwargs,
    ) -> tuple[list[namedtuple], tuple[str], tuple[str], str, Optional[dict[str, int]]]:
        """Generate experiences consisting of states, actions and rewards to
        train the agent. The generated values depends of the target algorithm.

        Parameters
        ----------
        n_experiences : int
            Number of experiences to generate.
        environment : environment
            Environment object of the problem.
        q_net : QNN
            Network for the prediction of all action-state values for a given
            state.
        device : Literal["cuda", "mps", "cpu"]
            Currently used device for training.
        epsilon : float
            Value of epsilon in epsilon greedy policy. With higher
            epsilon, more exploratory behaviour of the policy.
        initial_state : Literal["rand"], str
            State to initialize the generation of experiences from.
            If "rand", create a random initial state.
        initial_action : Optional[dict[str,int]], optional
            First action with label as key and index as value of experience
            generation. If None, select an action according to epsilon_greedy
            behaviour policy. By default, None.
        follow_next_action : bool, optional
            Select to store and follow a' along all experiencies generation.
            By default, False.
        decorrelated : bool, optional
            Select if experience samples are decorrelated.
            If True, they will be decorrelated.
            By default, False.

        ** kwargs
            step_count : bool, optional
                Select to have an steps counter.
                Specially usefull for those algorithms which do not have a natural
                terminal state, so it is implemented as an episode lenght.

        Returns
        -------
        experiences : Union[list[namedtuple[str, int, float, str]], list[namedtuple[str, int, float, str, int]]]
            List which contains each one of the experiences, composed by
            (state_label, action_index, reward_value, next_state_label).
            If follow_next_action, also include next_action.
        visited_states : tuple[srt]
            Tuple of visited states.
        reached_states : tuple[srt]
            Tuple of reached states, useful when the reward is computed as a
            function of the next state.
        last_next_state : str
            Last state s' visited. Useful to continue the trajectory of (s, a, r,
            s') generated.
        last_next_action : Optional[dict[str, int]]
            Last action a' performed. Useful to continue the sequence of
            generated experiences.

        See Also
        --------
        environment.py
            Where kwargs such as `step_count` will be applied as input.

        Notes
        -----
        * End of episode can be reached while experience generation. In this
          case, a restart of the environment will be done.
        * Decorrelated transitions are given through random sampling of states.
          Several authors recommend this practice.
        * We will skip transitions that starts at the terminal state, defined by
        the environment.
        """
        # if input inital state is a terminal state, require a new input
        if initial_state == environment.terminal_state:
            raise InputError("Initial state provided is a terminal state.")

        # if input inital action has more than one element, require the user to
        # select just one
        if exists(initial_action) and len(initial_action) > 1:
            raise InputError(
                "More than one initial action provided, please select just one."
            )

        # initialize first considered state for rand option.
        while initial_state in ["rand", environment.terminal_state]:
            initial_state = environment.random_state_label_choice()

        # set state label as initial state
        state_label = initial_state
        # set action as initial action
        action = initial_action

        # store generated experiences
        experiences = []
        visited_states = set()
        reached_states = set()
        for _ in range(n_experiences):
            # if decorrelated selected, randomly select next state
            if decorrelated:
                state_label = environment.random_state_label_choice()
                while state_label == environment.terminal_state:
                    state_label = environment.random_state_label_choice()

            # store visited states
            visited_states.add(state_label)

            # if action has been generated and selected to follow or provided as
            # input, follow it
            if exists(action):
                action_label, action_index = list(action.items())[0]

            # else, choose action with behaviour policy
            else:
                action_index, _, action_label = self._act(
                    "epsilon_greedy",
                    state_label,
                    q_net=q_net,
                    device=device,
                    epsilon=epsilon,
                )

            # observe response of the environment
            next_state_label, reward, end_episode = environment.step(
                state_label, action_label, **kwargs
            )
            # store reached states
            reached_states.add(next_state_label)

            # store the transition
            if not follow_next_action:
                experiences.append(
                    transition(state_label, action_index, reward, next_state_label)
                )

            # store sarsa transition if follow_next_action is selected
            if follow_next_action:
                # perform next action too
                next_action_index, _, next_action_label = self._act(
                    "epsilon_greedy",
                    next_state_label,
                    q_net=q_net,
                    device=device,
                    epsilon=epsilon,
                )
                next_action = {next_action_label: next_action_index}

                # store next action in transition
                experiences.append(
                    sarsa_transition(
                        state_label,
                        action_index,
                        reward,
                        next_state_label,
                        next_action,
                    )
                )

            # consider the end of the episode or continue from next state
            if end_episode:
                state_label = environment.reset()
            else:
                state_label = next_state_label

            # If follow_next_action is selected, force reset action to None if
            # end_episode has been reached. Otherwise, set action to None.
            if follow_next_action:
                if end_episode:
                    action = None
                else:
                    action = next_action

            # set action to None once its input has been employed, so we do not
            # get stuck in the initial action for follow_next_action = False
            else:
                action = None

        # store final visited state and next action
        last_next_state = state_label
        last_next_action = action

        return (
            experiences,
            visited_states,
            reached_states,
            last_next_state,
            last_next_action,
        )

    def _act(
        self,
        mode: Literal["greedy", "epsilon_greedy"],
        state_label: str,
        q_net: QNN,
        device: Literal["cuda", "mps", "cpu"],
        **kwargs,
    ) -> tuple[int, float, str]:
        """Returns the action that the agent takes given an state, i. e., the
        application of the policy.

        Parameters
        ----------
        mode : Literal["greedy", "epsilon_greedy"]
            Mode of acting.
            We can select:
                * "greedy" actions
                * "epsilon_greedy" actions
        state_label : str
            Current state label of the agent.
        q_net : QNN
            Network for the prediction of all action-state values for a given
            state.
        device : Literal["cuda", "mps", "cpu"]
            Currently used device for training.

        ** kwargs
            epsilon : float
                Value of epsilon in epsilon greedy policy. With higher
                epsilon, more exploratory behaviour of the policy.

        Returns
        -------
        action_idx, q_value, action_label : tuple[int, float, str]
            Index, value and label of the action taken by the agent.

        Warnings
        --------
        It is understood that the order of the output q_values from the `q_net`
        is the same that the order of the actions in `self.actions`. Else,
        returns will be inconsistent with the desired functionality.

        References
        ----------
        ..[1] https://pytorch.org/docs/stable/generated/torch.max.html#torch.max
        """
        # Transform state form label to list[int]
        state_list = str_to_tuple_or_list(state_label, to="list")

        # Obtain the action-state values for all actions from input `state`
        # It is necessary to set input as float32 so Pythorch does not return
        # us a `RuntimeError` due dtypes
        # Additionally, execute the forward pass at the same device we are using for
        # training to avoid a Pytorch `RuntimeError`
        q_values = q_net.forward(
            torch.from_numpy(np.array(state_list, dtype=np.float32)).to(device)
        )

        # -------- BASIC CHECKS --------
        # check if the number of outputs are the same than the number of actions
        if len(self.actions) != q_values.shape[0]:
            raise InputError(
                "Outputs of the `q_net` should correspond to the actions "
                "stored in `self.actions`, even respecting the order."
            )

        # -------- ACTION SELECTION --------
        # epsilon greedy behaviour
        if mode == "epsilon_greedy":
            try:
                epsilon = kwargs["epsilon"]
            except KeyError:
                raise InputError("Epsilon of epsilon greedy policy not provided.")

            rand = random.uniform(0, 1)

        # greedy behaviour
        elif mode == "greedy":
            # always get a random number above epsilon, so random choice is
            # never taken
            epsilon = 0
            rand = 1

        # select the action depending on a random number and epsilon value
        if epsilon > rand:
            action_idx = random.randint(0, len(self.actions) - 1)

        # select the action with a greedy policy
        else:
            # store the max action value
            max_val = torch.max(q_values)

            # store the max actions indexes
            max_action_idxs = [
                idx for idx, q_value in enumerate(q_values) if q_value == max_val
            ]

            # select randomly the action between actions which presents the
            # maximum value (so if there is a tie, `torch.max` do not take
            # always the first action) [2]
            action_idx = random.choice(max_action_idxs)

        # get action value and label with selected index
        action_val = q_values[action_idx]
        action_label = self.actions[action_idx]

        # check if the action seected is the maximum value action with greedy
        # behaviour
        if mode == "greedy":
            assert max_val == action_val

        return action_idx, action_val, action_label

    def _check_train_inputs(
        self,
        epsilon: float,
        lr: float,
        discount_rate: float,
        plot_learning_curves: bool,
        reward_curve_mode: list[str],
        reward_curve_steps_per_point: int,
    ):
        """Basic checks of `.train` inputs."""
        if epsilon > 1 or epsilon < 0:
            raise InputError("Epsilon must be a number between 0 and 1.")
        if lr > 1 or lr < 0:
            raise InputError("Learnig rate must be a number between 0 and 1.")
        if discount_rate > 1 or discount_rate < 0:
            raise InputError("Discount rate must be a number between 0 and 1.")
        if plot_learning_curves and reward_curve_steps_per_point is None:
            warnings.warn(
                "In order to plot reward curves, `reward_curve_steps_per_point` must be other than None."
            )
        if not set(reward_curve_mode) <= set(
            ["return", "last_reward", "best_reward", "best_htc"]
        ):
            raise InputError(
                "Invalid reward curve mode. Please, select 'return', 'last_reward', 'best_reward' or 'best_htc'."
            )

    def train(
        self,
        device: Literal["cuda", "mps", "cpu"],
        environment: environment,
        discount_rate: float = 0.99,
        lr: float = 0.1,
        epsilon: float = 1.0,
        batch_size: int = 64,
        max_steps: int = np.inf,
        tol_loss: float = 0.0,
        plot_learning_curves: bool = True,
        save_q_net: bool = True,
        **kwargs,
    ):
        """Train a neural network to predict action-state values from input
        states.

        Parameters
        ----------
        device : Literal["cuda", "mps", "cpu"]
            Currently used device for training. Used as an `act` method input.
        environment : environment
            Environment object of the problem.
        discount_rate : float, optional
            Discount rate factor for Reinforcement Learning algorithm.
            By default, 0.99
        lr : float, optional
            Value of the learning rate.
            By default, 0.1
        epsilon : float, optional
            Initial value of epsilon for epsilon greedy policies.
            By default, 1.0
        batch_size : int, optional
            Size of the batch for training the neural network for the prediction
            of the action-state values.
            By default, 64
        max_steps : int, optional
            Maximum number of steps to iterate policy evaluation.
            By default, np.inf
        tol_loss : float, optional
            Tolerance to consider action values have converged.
            By default, 0.0
        plot_learning_curves : bool, optional
            Select to plot learning curves or not.
            By defaul, True
        save_q_net : bool, optinal
            Select to save the action-state values network.
            By default, True.

        Other Parameters
        ----------------
        **kwargs
            reduce_eps : float, optional
                Ammount to reduce epsilon at each iteration.
                By default, 1e-3
            min_eps : float, optional
                Minimum value of epsilon.
                By default, 0.0

            reduce_perc_lr : float, optional
                Percentage to reduce learning rate at each iteration.
                By default, 0.0001
            min_lr : float, optional
                Minimum value of learning rate.
                By default, 0.0

            reward_curve_mode : list[Literal["return", "last_reward", "best_reward", "best_htc"]], optional
                List with the selection of the rewards to record at `reward_curve`:
                    * return : plot the return of the start state for each
                        simulation.
                    * last_reward : plot the reward of the last step of each
                        simulation.
                    * best_reward : plot the best reward seen during all the
                      training.
                    * best_htc : plot the best htc value (higher htc value) seen
                      during all the training.
                By default, ["return", "last_reward", "best_reward", "best_htc"]
            reward_curve_steps_per_point : Optional[int], optional
                Select the number of steps for each one of the simulated
                episodes created to plot each point of the reward curve.
                If None, DO NOT RECORD any reward learning curve.
                By default, 30.

            episode_start_state : [Literal["rand", "initial"], str], optional
                State to initialize the episodes from.
                If "rand", create a random initial state each time.
                If "initial", start from environment start state.
                By default, "initial".

            decorrelated : bool, optional
                Select if batch samples are decorrelated.
                If True, they will be decorrelated.
                By default, False.
            step_count : bool, optional
                Select to have a steps counter.
                Specially usefull for those algorithms which do not have a natural
                terminal state, so it is implemented as an episode lenght.
                By default, False, so step count is inactive.

            debug_counter : int, optional
                Select every how many counts logging debug will be given.
                By default, 1.

            seed : int, optional
                Seed number for random and torch modules. If None, do not fix
                any seed.
                By default, None.

        Warns
        -----
        * If `plot_learning_curves` is selected but
          `reward_curve_steps_per_point` is None.
        * If `reward_curve_steps_per_point` is set to other than None.

        Notes
        -----
        * Currently, one pair action-state is being updated for each individual
          sample of the batch.
        * Notice that `max_steps` is not related to the number of steps per
          episode. For this, we have environment step count. In addition,
          the number of samples to update the network at each step is determined
          by batch parameter.
          Meanwhile, tabular method is trained for a number of episodes composed
          by its number of steps.

        References
        ----------
        .. [1] https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
        .. [2] https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py
        .. [3] https://stackoverflow.com/questions/48324152/how-to-change-the-learning-rate-of-an-optimizer-at-any-given-moment-no-lr-sched
        """
        # fix seed
        seed = kwargs.get("seed", None)
        if exists(seed):
            random.seed(seed)
            torch.manual_seed(seed)

        # info
        logging.info(f"Training the agent with {self.algorithm} algorithm...")

        # ------ KWARGS ------
        reduce_eps = kwargs.get("reduce_eps", 1e-3)
        min_eps = kwargs.get("min_eps", 0.0)

        reduce_perc_lr = kwargs.get("reduce_perc_lr", 0.0001)
        min_lr = kwargs.get("min_lr", 0.0)

        reward_curve_mode = kwargs.get(
            "reward_curve_mode", ["return", "last_reward", "best_reward", "best_htc"]
        )
        reward_curve_steps_per_point = kwargs.get("reward_curve_steps_per_point", 30)

        episode_start_state = kwargs.get("episode_start_state", "initial")

        decorrelated = kwargs.get("decorrelated", False)
        step_count = kwargs.get("step_count", False)

        debug_counter = kwargs.get("debug_counter", 1)

        # basic checks of input values
        self._check_train_inputs(
            epsilon,
            lr,
            discount_rate,
            plot_learning_curves,
            reward_curve_mode,
            reward_curve_steps_per_point,
        )

        # -------------------------------------------------------------------------
        # Step 0: define the NN of the Q function
        # -------------------------------------------------------------------------
        # Input dim is the number of aspects that define our state.
        # Take into account that, in our implementation, the state will be given
        # by a str with a collection of numbers.
        in_dim = len(str_to_tuple_or_list(environment.start_state, to="list"))

        # Output dim will be given by the number of possible actions for each
        # state, i. e., number of possible actions.
        out_dim = len(self.actions)

        # create the neural network
        q_net = QNN(
            in_dim,
            out_dim,
            hidden_layers=self.hidden_layers,
            hidden_neur=self.hidden_neur,
        )
        # and select its training hardware
        q_net = q_net.to(device)

        # set the optimizer
        optimizer = optim.Adam(q_net.parameters(), lr=lr)

        # set the loss
        loss_L1 = nn.L1Loss()

        # -------------------------------------------------------------------------
        # Step 1: perform the training of the system
        # -------------------------------------------------------------------------
        # ------ INITIALIZATION ------
        # initialize epsilon, amount of loss and storage of best reward seen
        # along all the training
        step = 1
        loss = np.inf
        training_best_reward = -np.inf
        training_best_htc = -np.inf

        # initialize learning curves
        loss_curve = learning_curve()
        reward_curves = []
        if exists(reward_curve_steps_per_point):
            for _ in reward_curve_mode:
                reward_curves.append(learning_curve())

            warnings.warn(
                "In order to save `reward_curves`, an additional simulation at "
                "each step of the algorithm is made. "
                "Please, consider if this type of plot is necessary. "
                "If not, set `reward_curve_steps_per_point` to None."
            )
        known_states_list = []
        max_htc_vs_known_state = learning_curve()

        # store visited states
        visited_states = set()

        # define episode start with reset method
        episode_start_state = environment.reset(episode_start_state)

        # initialize batch state
        initial_batch_state = episode_start_state

        # select to store a' only for Sarsa algorithm
        follow_next_action = True if self.algorithm == "Sarsa" else False
        initial_batch_action = None

        # ------ ALGORITHM ------
        # loop during a detemined number of steps or until convergence
        while step <= max_steps and tol_loss < loss:
            # output step info if requested
            if self.verbose and step % debug_counter == 0:
                logging.debug(f"Computing step number {step}...")

            # -------------------------------------------------------------------------
            # Step 1.1: generate for each step the experiences for training and
            # store them in a batch
            # -------------------------------------------------------------------------
            (
                batch,
                batch_visited_states,
                _,
                last_batch_state,
                last_batch_action,
            ) = self._experience_generation(
                batch_size,
                q_net,
                environment,
                device,
                epsilon,
                initial_batch_state,
                initial_batch_action,
                follow_next_action,
                decorrelated,
                step_count=step_count,
            )
            initial_batch_state = last_batch_state
            initial_batch_action = last_batch_action

            # store unique visited states with the usage of set
            for state in batch_visited_states:
                visited_states.add(state)

            # -------------------------------------------------------------------------
            # Step 1.2: use the batch experiences to get several pairs
            # estimation and target q values
            # -------------------------------------------------------------------------
            batch_estimations = []
            batch_targets = []
            for experience in batch:
                # Transform state form label to list[int]
                state_list = str_to_tuple_or_list(experience.state, to="list")

                # obtain ALL the q value estimations of the net for state
                # It is necessary to set input as float32 so Pythorch does not
                # return us a `RuntimeError` due dtypes
                # Additionally, execute the forward pass at the same device we
                # are using for training to avoid a Pytorch `RuntimeError`
                estimated_q_values = q_net.forward(
                    torch.from_numpy(np.array(state_list, dtype=np.float32)).to(device)
                )
                batch_estimations.append(estimated_q_values[experience.action_idx])

                # do not follow the gradient for obtained target values
                with torch.no_grad():
                    # take the q value for the next action-state pair with
                    # target policy (Sarsa, Q-learning)
                    if self.algorithm == "Sarsa":
                        # retrieve from experience the next action taken with
                        # behaviour policy and compute its q_value
                        # 1- obtain the action-state values for all actions from `next_state`
                        next_state_list = str_to_tuple_or_list(
                            experience.next_state, to="list"
                        )
                        next_q_values = q_net.forward(
                            torch.from_numpy(
                                np.array(next_state_list, dtype=np.float32)
                            ).to(device)
                        )
                        # 2- get action value with selected next_action index
                        next_q_value = next_q_values[
                            list(experience.next_action.values())[0]
                        ]

                    elif self.algorithm == "Q-learning":
                        # greedy action as target behaviour for Q-learning
                        _, next_q_value, _ = self._act(
                            "greedy",
                            experience.next_state,
                            q_net=q_net,
                            device=device,
                        )

                    # update the target action values
                    if self.algorithm in ["Sarsa", "Q-learning"]:
                        # use the action value of the next action (Sarsa, Q-learning)
                        # Set reward tensor to avoid a Pytorch `RuntimeError`
                        target_q_value = torch.squeeze(
                            torch.Tensor([experience.reward]).to(device)
                            + discount_rate * next_q_value
                        )

                    batch_targets.append(target_q_value)

                # Update stored unique reached states. If an state has been
                # already reached (we know its contribution to the reward), max
                # reward will be the same, as we are not discovering a new
                # contribution.
                if experience.next_state not in known_states_list:
                    (
                        updated_training_best_htc,
                        training_best_reward,
                        known_states_list,
                        max_htc_vs_known_state,
                    ) = self._update_max_htc_vs_known_state(
                        experience,
                        environment,
                        known_states_list,
                        training_best_htc,
                        training_best_reward,
                        max_htc_vs_known_state,
                        step,
                    )
                    # update `training_best_htc` after checking its increase (it
                    # should not change if it has not increased)
                    training_best_htc = updated_training_best_htc

            # get the loss of the action value to update in the q net
            # detach indicates to not follow the gradient for the target, as it
            # implies the use of the q net too
            loss = loss_L1(
                torch.stack(batch_estimations), torch.stack(batch_targets).detach()
            )

            # Update the network
            optimizer.zero_grad()  # Reset the gradients an usual practice
            loss.backward()  # backpropagation Gradient descent
            optimizer.step()  # update network weights

            # update loss curve
            loss_curve.update(loss.item(), step, learning_rate=lr, epsilon=epsilon)

            # update reward curves
            if exists(reward_curve_steps_per_point):
                self._update_reward_curves(
                    environment,
                    q_net,
                    episode_start_state,
                    step,
                    device,
                    reward_curves,
                    reward_curve_mode,
                    reward_curve_steps_per_point,
                    lr,
                    epsilon,
                    training_best_reward,
                    training_best_htc,
                )

            # reduction of epsilon at each episode, with a min value of min_eps
            epsilon -= reduce_eps
            epsilon = max(epsilon, min_eps)

            # reduction of learning rate at each episode, with a min value of
            # min_lr
            lr -= lr * reduce_perc_lr / 100
            lr = max(lr, min_lr)
            # update optimizer with next learning rate [3]
            for g in optimizer.param_groups:
                g["lr"] = lr

            # update the step number
            step += 1

            # output convergence info if requested
            if self.verbose and step % debug_counter == 0:
                logging.debug(f"The loss in step {step} has been {loss:.3}")

        # -------------------------------------------------------------------------
        # Step 2: save relevant training products
        # -------------------------------------------------------------------------
        self.q_net = q_net
        self.visited_states = visited_states
        self.known_states_list = known_states_list

        self.loss_curve = loss_curve
        self.max_htc_vs_known_state = max_htc_vs_known_state
        if exists(reward_curve_steps_per_point):
            for idx, reward_curve_mode_ in enumerate(reward_curve_mode):
                if reward_curve_mode_ == "return":
                    self.reward_curve_return = reward_curves[idx]
                elif reward_curve_mode_ == "last_reward":
                    self.reward_curve_last_reward = reward_curves[idx]
                elif reward_curve_mode_ == "best_reward":
                    self.reward_curve_best_reward = reward_curves[idx]
                elif reward_curve_mode_ == "best_htc":
                    self.reward_curve_best_htc = reward_curves[idx]

        if save_q_net:  # [1]
            torch.save(
                q_net.state_dict(),
                f"./data/{self.save_folder}/q_net_{in_dim}_inputs.pt",
            )

        # -------------------------------------------------------------------------
        # Step 3: plot relevant data and save their figures and objects
        # -------------------------------------------------------------------------
        if plot_learning_curves:
            self._plot_learning_curves(
                environment.n_layers,
                loss_curve,
                max_htc_vs_known_state,
                reward_curves,
                reward_curve_mode,
                reward_reduction_factor=environment.reward_reduction_factor,
            )

    def load_net(
        self,
        in_dim: int,
        device: Literal["cuda", "mps", "cpu"],
    ):
        # Input dim is the number of aspects that define our state.
        # Output dim will be given by the number of possible actions for each
        # state, i. e., number of possible actions.
        out_dim = len(self.actions)

        # create the neural network object
        q_net = QNN(
            in_dim,
            out_dim,
            hidden_layers=self.hidden_layers,
            hidden_neur=self.hidden_neur,
        )

        # load weights and biases
        w_and_b = torch.load(f"./data/{self.save_folder}/q_net_{in_dim}_inputs.pt")
        # load weights and biases into created neural network object
        # employ `w_and_b` for the net before any operation to avoid consuming
        # the iterable
        q_net.load_state_dict(w_and_b)
        # load newtork into selected device to ensure we are employing always
        # the same device
        self.q_net = q_net.to(device)

        # output info about network number of layers and neurons inside them
        inputs_nd_n_neurons = [
            (weights.shape[1], weights.shape[0])
            for key, weights in w_and_b.items()
            if "weight" in key
        ]
        logging.info(
            "Loaded neural network with the following architecture (input layer not included):"
        )
        for n_layer, shapes in enumerate(inputs_nd_n_neurons):
            logging.info(
                f"\tLayer {n_layer} with {shapes[0]} inputs and {shapes[1]} neurons"
            )

    def _update_max_htc_vs_known_state(
        self,
        experience: namedtuple,
        environment: environment,
        known_states_list: list[str],
        discovered_best_htc: float,
        discovered_best_reward: float,
        max_htc_vs_known_state: learning_curve,
        step: Optional[int] = None,
    ) -> tuple[float, float, list[str], learning_curve]:
        """Update `max_htc_vs_known_state` learning curve and retrieve
        `training_best_htc` and `training_best_reward`, in addition to update
        known states list.

        Parameters
        ----------
        experience: namedtuple
            Experience of (s, a, r, s') to use in order to update the learning
            curve and known_states_list.
        environment: environment
            Environment of the simulation, used to:
                * Retrieve `htc_value` of the experience.
        known_states_list: list[str]
            List of known states, i.e., for which htc has been computed.
        discovered_best_htc: float
            Best htc seen so far.
        discovered_best_reward: float
            Best reward seen so far.
        max_htc_vs_known_state: learning_curve
            Learning curve storing max htc seen as a function of number of known
            states. Updated inside this utility.
        step: Optional[int], optional
            Training step, for informative purposes in case a better htc is
            found. If None, do not display this information. By default, None.

        Returns
        -------
        discovered_best_htc : float
            Best htc discovered.
        discovered_best_reward : float
            Best reward of the RL problem discovered.
        known_states_list : list[str]
            List of known states. Remember known state is not the same as
            visited state, it is defined as the states of which htc is known.
        max_htc_vs_known_state : learning_curve
            Updated learning curve with discovered max htc vs number of known
            states.

        Raises
        ------
        InputError
            In `next_state` stored in input `experience` already is in
            `known_states_list`.

        Warnings
        --------
        * Because of the use case of this method, htc value of s' of input
          experience is assumed to be stored in the environment.
        * For HTC problem, `known_states_list` is equivalente to the storage of
          all `next_state`. This is due to the fact that the reward for the HTC
          problem is only defined based on the HTC of the next state.
        """
        if experience.next_state in known_states_list:
            raise InputError("Input next state already known.")

        known_states_list.append(experience.next_state)

        # store the best htc, reward and state label found during training
        experience_htc = environment.htc_values[experience.next_state]

        if experience_htc > discovered_best_htc:
            discovered_best_htc = experience_htc
            discovered_best_reward = experience.reward

            if step is not None:
                logging.info(
                    f"Better state found at step {step} and state number {len(known_states_list)}: {experience.next_state} ({experience_htc})"
                )

        # save items for max htc - state curve
        max_htc_vs_known_state.update(discovered_best_htc, len(known_states_list))

        return (
            discovered_best_htc,
            discovered_best_reward,
            known_states_list,
            max_htc_vs_known_state,
        )

    def _update_reward_curves(
        self,
        environment: environment,
        q_net: QNN,
        episode_start_state: str,
        step: int,
        device: Literal["cuda", "mps", "cpu"],
        reward_curves: list[learning_curve],
        reward_curve_mode: list[str],
        steps_per_point: int,
        lr: Optional[float] = None,
        epsilon: Optional[float] = None,
        training_best_reward: Optional[float] = None,
        training_best_htc: Optional[float] = None,
    ):
        """Update selected reward curves in `reward_curve_mode` for each step.
        For that, we perform a greedy simulation along selected `steps_per_point`
        whith the current state of the `q_net`.

        Parameters
        ----------
        environment: environment
            Environment of the simulation, deep copied to perform the greedy
            simulation.
        q_net: QNN
            Q-network with which make the simulation.
        episode_start_state: str
            Start state of an episode. Used as starting point of the simulation.
        step: int
            Step of the training, x-axis of reward curves.
        device: Literal["cuda", "mps", "cpu"]
            Currently used device for training. Used as an `act` method input.
        reward_curves: list[learning_curve]
            Rewards curves to update.
        reward_curve_mode: list[str]
            Reward curve modes to retrieve in order to update input reward
            curves.
        steps_per_point: int
            Number of steps of the simulation.
        lr: Optional[float], optional
            Learning reate value. If None, do not represent it in reward curves.
            By default, None.
        epsilon: Optional[float], optional
            Epsilon value. If None, do not represent it in reward curves. By
            default, None.
        training_best_reward: Optional[float], optional
            Training best reward for input step. Necessary in order to update
            best_reward learning curve.
        training_best_htc: Optional[float], optional
            Training best htc for input step. Necessary in order to update
            best_htc learning curve.

        Warnings
        --------
        * It is assumed `reward_curve_mode` matches input `reward_curves` to
          update. TODO In future versions these inputs will be unified in order
          to assure it and secure the correct name/object assignation.

        References
        ----------
        .. [1] https://docs.python.org/3/library/copy.html
        """
        if len(reward_curves) != len(reward_curve_mode):
            raise warnings.warn(
                "Number of input reward curves does not match the number of "
                f"reward curve modes. Only {reward_curve_mode} will be updated."
            )

        if "best_reward" in reward_curve_mode and training_best_reward is None:
            raise InputError(
                "`training_best_reward` is required as input in order to update "
                " best_reward learning curve."
            )

        if "best_htc" in reward_curve_mode and training_best_htc is None:
            raise InputError(
                "`training_best_htc` is required as input in order to update "
                " best_htc learning curve."
            )

        # make a small simulation to get the rewards of the net for an episode
        overall_return, last_reward, _, _, _ = self.greedy_simulation(
            q_net=q_net,
            environment=copy.deepcopy(environment),  # [1]
            start_state=episode_start_state,
            steps=steps_per_point,
            device=device,
        )
        for idx, reward_curve_mode_ in enumerate(reward_curve_mode):
            if reward_curve_mode_ == "return":
                step_reward = overall_return
            elif reward_curve_mode_ == "last_reward":
                step_reward = last_reward
            elif reward_curve_mode_ == "best_reward":
                step_reward = training_best_reward
            elif reward_curve_mode_ == "best_htc":
                step_reward = training_best_htc
            # update learning curves
            reward_curves[idx].update(step_reward, step, lr, epsilon)

    def _plot_learning_curves(
        self,
        n_layers: int,
        loss_curve: learning_curve,
        max_htc_vs_known_state: learning_curve,
        reward_curves: list[learning_curve],
        reward_curve_mode: list[str],
        reward_reduction_factor: Optional[str] = None,
    ):
        """Plot learning curves adapted to `DRL_agent` outputs.

        Warnings
        --------
        * It is assumed `reward_curve_mode` matches input `reward_curves` to
          update. TODO In future versions these inputs will be unified in order
          to assure it and secure the correct name/object assignation.
        """
        if len(reward_curves) != len(reward_curve_mode):
            raise warnings.warn(
                "Number of input reward curves does not match the number of "
                f"reward curve modes. Only {reward_curve_mode} will be updated."
            )
        # obtain units label
        units_label = HTC_units_label(reward_reduction_factor)

        loss_curve.plot(
            title="",
            xlabel="Training step",
            ylabel="MAE loss",
            plot_epsilon=True,
            plot_lr=False,
            save_path=f"./img/{self.save_folder}/Loss_learning_curve_{n_layers}.png",
        )

        max_htc_vs_known_state.plot(
            title="",
            xlabel="Found states",
            ylabel="Largest HTC ($\mathdefault{W/m^2K}$)",
            plot_epsilon=False,
            plot_lr=False,
            save_path=f"./img/{self.save_folder}/Max_htc_per_known_state_{n_layers}.png",
        )

        if len(reward_curves) != 0:
            for idx, reward_curve_mode_ in enumerate(reward_curve_mode):
                if reward_curve_mode_ == "return":
                    reward_ylabel = f"Return ({units_label})"
                    sufix = "return"
                elif reward_curve_mode_ == "last_reward":
                    reward_ylabel = f"Last reward of greedy simulation ({units_label})"
                    sufix = "last_reward"
                elif reward_curve_mode_ == "best_reward":
                    reward_ylabel = f"Maximum reward ({units_label})"
                    sufix = "best_reward"
                elif reward_curve_mode_ == "best_htc":
                    reward_ylabel = f"Maximum heat transfer coefficient ({units_label})"
                    sufix = "best_htc"

                reward_curves[idx].plot(
                    title="",
                    xlabel="Training step",
                    ylabel=reward_ylabel,
                    plot_epsilon=True,
                    plot_lr=False,
                    y_divisor=(
                        reward_reduction_factor
                        if reward_curve_mode_ == "best_htc"
                        else None
                    ),
                    save_path=f"./img/{self.save_folder}/Reward_learning_curve_{n_layers}_{sufix}.png",
                )


class DQN_agent(DRL_agent):
    def __init__(
        self,
        algorithm: Literal["Q-learning", "double_Q-learning"],
        actions: Iterable[str],
        save_folder: str = "DRL_results",
        verbose: bool = False,
        **kwargs,
    ):
        """Deep Q-Networks algorithms.

        Implemented modification with respect plain Deep Q-learning in
        `DRL_agent`:
         - Memory replay (always active)
         - Target network
         - Double estimation

        Notes
        -----
        `Target network` and `Double estimation` can be either selected or
        deactivated through `target_estimation_mode` at `train` method.
        By default `Double estimation` is selected.
        """

        # check for some errors
        if algorithm not in ["Q-learning", "double_Q-learning"]:
            raise InputError(
                "Deep Q-learning agent can only be used for Q-learning algorithms."
                f"Input algorithm was {algorithm}."
            )

        # inherit from parent class
        super().__init__(
            algorithm,
            actions,
            save_folder,
            verbose,
            **kwargs,
        )

    def train(
        self,
        device: Literal["cuda", "mps", "cpu"],
        environment: environment,
        discount_rate: float = 0.99,
        lr: float = 0.1,
        epsilon: float = 1.0,
        batch_size: int = 64,
        max_steps: int = np.inf,
        tol_loss: float = 0.0,
        plot_learning_curves: bool = True,
        save_q_net: bool = True,
        memory_size: int = 10000,
        n_batch_per_step: int = 4,
        n_new_experiences_per_step: int = 1,
        target_estimation_mode: Literal[
            "regular", "target network", "double"
        ] = "double",
        n_steps_for_target_net_update: int = 1000,
        **kwargs,
    ):
        """Train a neural network to predict action-state values from input
        states.

        Parameters
        ----------------
        memory_size : int, optional
            Maximum length of memory replay.
            By default, 10000 [2]
        n_batch_per_step : int, optional
            Number of batches to retrieve from memory for each training step.
            By default, 4 [2]
        n_new_experiences_per_step: int, optional
            Number of new experiences to store each training step.
            By default, 1. [1]
        target_estimation_mode : Literal["regular", "target network", "double"], optional
            Select if we train Deep Q-Networks (DQN) with the estimation of the
            Q value forming the train target with the:
                * "regular" : Network that is being trained (training network).
                * "target network" : Target network (adds the use of such network).
                * "double" : The combination of the training network and target
                  network (technique known as double DQN).
            By default, "double".
        n_steps_for_target_net_update : int, optional
            Number of steps to wait to update target network weights with those
            of the training Q-network. Only necessary if
            `target_estimation_mode` is "target network" or "double".
            By default, 1000. [1]

        See Also
        --------
        DRL_agent : parent class.
            Most of input parameters are described at its `.train` method
            dosctring.

        Notes
        -----
        * Currently, one pair action-state is being updated for each individual
          sample of the batch.
        * Notice that `max_steps` is not related to the number of steps per
          episode. For this, we have environment step count. In addition, the
          number of samples to update the network at each step is determined by
          batch parameter.
          Meanwhile, tabular method is trained for a number of episodes composed
          by its number of steps.
        * Notice that exist two `best_htc` trackings: `memory_best_htc` and
          `training_best_htc`. Learning curve `max_htc_vs_known_state` is only
          refered to the first one, `memory_best_htc`.

        References
        ----------
        .. [1] https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        .. [2] Fundations of Deep Reinforcement Learning, Laura Graesser and Wah Loon Keng
        .. [3] https://docs.python.org/3/library/copy.html
        .. [4] https://iamholumeedey007.medium.com/copy-deepcopy-vs-clone-in-pytorch-e5b951b0cea3
        """
        # fix seed
        seed = kwargs.get("seed", None)
        if exists(seed):
            random.seed(seed)
            torch.manual_seed(seed)

        # info
        logging.info(f"Training the agent with {self.algorithm} algorithm...")

        # ------ KWARGS ------
        reduce_eps = kwargs.get("reduce_eps", 1e-3)
        min_eps = kwargs.get("min_eps", 0.0)

        reduce_perc_lr = kwargs.get("reduce_perc_lr", 1e-4)
        min_lr = kwargs.get("min_lr", 0.0)

        reward_curve_mode = kwargs.get(
            "reward_curve_mode", ["return", "last_reward", "best_reward", "best_htc"]
        )
        reward_curve_steps_per_point = kwargs.get("reward_curve_steps_per_point", 30)
        episode_start_state = kwargs.get("episode_start_state", "initial")

        decorrelated = kwargs.get("decorrelated", False)
        step_count = kwargs.get("step_count", False)

        debug_counter = kwargs.get("debug_counter", 1)

        # define episode start with reset method
        episode_start_state = environment.reset(episode_start_state)

        # initialize batch state
        initial_batch_state = episode_start_state

        # basic checks of input values
        if epsilon > 1 or epsilon < 0:
            raise InputError("Epsilon must be a number between 0 and 1.")
        if lr > 1 or lr < 0:
            raise InputError("Learnig rate must be a number between 0 and 1.")
        if discount_rate > 1 or discount_rate < 0:
            raise InputError("Discount rate must be a number between 0 and 1.")
        if plot_learning_curves and reward_curve_steps_per_point is None:
            warnings.warn(
                "In order to plot reward curves, `reward_curve_steps_per_point` must be other than None."
            )
        if not set(reward_curve_mode) <= set(
            ["return", "last_reward", "best_reward", "best_htc"]
        ):
            raise InputError(
                "Invalid reward curve mode. Please, select 'return', 'last_reward', 'best_reward' or 'best_htc'."
            )
        if target_estimation_mode not in ["regular", "target network", "double"]:
            raise InputError(
                "Invalid `target estimation mode`. Please, select 'regular', 'target network' or 'double'."
            )

        # -------------------------------------------------------------------------
        # Step 0: define the NN of the Q function
        # -------------------------------------------------------------------------
        # Input dim is the number of aspects that define our state.
        # Take into account that, in our implementation, the state will be given
        # by a str with a collection of numbers.
        in_dim = len(str_to_tuple_or_list(environment.start_state, to="list"))

        # Output dim will be given by the number of possible actions for each
        # state, i. e., number of possible actions.
        out_dim = len(self.actions)

        # create the neural network
        q_net = QNN(
            in_dim,
            out_dim,
            hidden_layers=self.hidden_layers,
            hidden_neur=self.hidden_neur,
        )
        # and select its training hardware
        q_net = q_net.to(device)

        # set the optimizer
        optimizer = optim.Adam(q_net.parameters(), lr=lr)

        # set the loss
        loss_L1 = nn.L1Loss()

        if target_estimation_mode in ["target network", "double"]:
            # create the target neural network [4]
            q_net_target = copy.deepcopy(q_net)
            # initialize target network frecuency counter
            target_net_counter = 0
        # -------------------------------------------------------------------------
        # Step 1: perform the training of the neural network
        # -------------------------------------------------------------------------
        # ------ INITIALIZATION ------
        # initialize step number, amount of loss, memory of experiences,
        # visited_states storage, storage of best reward seen along all the
        # training, storage of known states and storage of the higher htc
        # obtained in memory experiences
        step = 1
        loss = np.inf
        replay_memory = ReplayMemory(
            memory_size,
            self,
            n_experiences=batch_size,  # agent._experience_generation kwargs
            q_net=q_net,
            environment=environment,
            device=device,
            epsilon=epsilon,
            initial_state=initial_batch_state,
            follow_next_action=False,
            decorrelated=decorrelated,
            step_count=step_count,
        )
        visited_states = set()
        training_best_reward = -np.inf
        known_states_list = []
        memory_best_htc = -np.inf
        # initialize best htc value
        training_best_htc = -np.inf

        # initialize learning curves
        loss_curve = learning_curve()
        reward_curves = []
        if exists(reward_curve_steps_per_point):
            for _ in reward_curve_mode:
                reward_curves.append(learning_curve())

            warnings.warn(
                "In order to save `reward_curves`, an additional simulation at "
                "each step of the algorithm is made. "
                "Please, consider if this type of plot is necessary. "
                "If not, set `reward_curve_steps_per_point` to None."
            )
        max_htc_vs_known_state = learning_curve()

        # store explored states already gathered in memory replay during its
        # initialization and update `max_htc_vs_known_state`
        for experience in replay_memory.memory:
            if experience.next_state not in known_states_list:
                (memory_best_htc, _, known_states_list, max_htc_vs_known_state) = (
                    self._update_max_htc_vs_known_state(
                        experience,
                        environment,
                        known_states_list,
                        memory_best_htc,
                        _,
                        max_htc_vs_known_state,
                        step,
                    )
                )

        # ------ ALGORITHM ------
        # loop during a detemined number of steps or until convergence
        while step <= max_steps and tol_loss < loss:
            # output step info if requested
            if self.verbose and step % debug_counter == 0:
                logging.debug(f"Computing step number {step}...")

            # ----------------------------------------------------
            # Step 1.0: store new experiences in memory replay
            # ----------------------------------------------------
            # generate selected number of new experiences
            # continue experience storage from the state s' of the last experience
            new_experiences, _, _, _, _ = self._experience_generation(
                n_experiences=n_new_experiences_per_step,
                q_net=q_net,
                environment=environment,
                device=device,
                epsilon=epsilon,
                initial_state=replay_memory.memory[-1].next_state,
                follow_next_action=False,
                decorrelated=decorrelated,
                step_count=step_count,
            )

            # store the transition
            replay_memory.push(new_experiences)

            # Store unique known states and max htc when a new experience is
            # generated. If an state is known (we know its htc), max htc will
            # not be modified, as we are not discovering a new htc.
            # Best htc during MEMORY GENERATION.
            for experience in new_experiences:
                if experience.next_state not in known_states_list:
                    (memory_best_htc, _, known_states_list, max_htc_vs_known_state) = (
                        self._update_max_htc_vs_known_state(
                            experience,
                            environment,
                            known_states_list,
                            memory_best_htc,
                            _,
                            max_htc_vs_known_state,
                            step,
                        )
                    )

            for _ in range(n_batch_per_step):
                # -------------------------------------------------------------------------
                # Step 1.1: retrieve desired number of experiences and store them
                # as a batch [1]
                # -------------------------------------------------------------------------
                batch = replay_memory.sample(batch_size)

                # -------------------------------------------------------------------------
                # Step 1.2: use the batch experiences to get several pairs
                # estimation and target q values
                # -------------------------------------------------------------------------
                batch_estimations = []
                batch_targets = []
                for experience in batch:
                    # store unique visited states during the network training
                    # with the usage of set
                    visited_states.add(experience.state)

                    # Transform state form label to list[int]
                    state_list = str_to_tuple_or_list(experience.state, to="list")

                    # obtain ALL the q value estimations of the net for state
                    # It is necessary to set input as float32 so Pythorch does not
                    # return us a `RuntimeError` due dtypes
                    # Additionally, execute the forward pass at the same device we
                    # are using for training to avoid a Pytorch `RuntimeError`

                    estimated_q_values = q_net.forward(
                        torch.from_numpy(np.array(state_list, dtype=np.float32)).to(
                            device
                        )
                    )
                    batch_estimations.append(estimated_q_values[experience.action_idx])

                    # do not follow the gradient for obtained target values
                    with torch.no_grad():
                        if target_estimation_mode in ["regular", "double"]:
                            network_for_target_estimation = q_net

                        elif target_estimation_mode == "target network":
                            network_for_target_estimation = q_net_target

                        # greedy action as target behaviour for Q-learning
                        next_action_idx, next_q_value, _ = self._act(
                            "greedy",
                            experience.next_state,
                            q_net=network_for_target_estimation,
                            device=device,
                        )

                        # double DQN: take the q value from target network with
                        # action obtained from trained network
                        if target_estimation_mode == "double":
                            # next state
                            next_state_list = str_to_tuple_or_list(
                                experience.next_state, to="list"
                            )
                            # all `q_net_target` q values
                            target_net_q_values = q_net_target.forward(
                                torch.from_numpy(
                                    np.array(next_state_list, dtype=np.float32)
                                ).to(device)
                            )
                            # select next q value form target network with
                            # greedy action from trained network
                            next_q_value = target_net_q_values[next_action_idx]

                        # update the target action values
                        # use the action value of the next action (Sarsa, Q-learning)
                        # Set reward tensor to avoid a Pytorch `RuntimeError`
                        target_q_value = torch.squeeze(
                            torch.Tensor([experience.reward]).to(device)
                            + discount_rate * next_q_value
                        )

                        batch_targets.append(target_q_value)

                    # store the best htc, reward and state label found during
                    # TRAINING
                    experience_htc = environment.htc_values[experience.next_state]
                    if experience_htc > training_best_htc:
                        training_best_htc = experience_htc
                        training_best_reward = experience.reward
                        logging.debug(
                            f"Better state used for training at step {step} and visited state number {len(visited_states)}: {experience.next_state} ({experience_htc})"
                        )

                # get the loss of the action value to update in the q net
                # detach indicates to not follow the gradient for the target, as it
                # implies the use of the q net too
                loss = loss_L1(
                    torch.stack(batch_estimations), torch.stack(batch_targets).detach()
                )

                # Update the network
                optimizer.zero_grad()  # Reset the gradients as usual practice
                loss.backward()  # backpropagation Gradient descent
                optimizer.step()  # update network weights

            # update loss curve when all batches per step have been processed
            loss_curve.update(loss.item(), step, learning_rate=lr, epsilon=epsilon)

            # update target network weights if `n_steps_for_target_net_update`
            # is reached
            if target_estimation_mode in ["target network", "double"]:
                target_net_counter += 1
                if target_net_counter == n_steps_for_target_net_update:
                    # copy q_net weight into target q_net
                    q_net_target.copy_weights(q_net)
                    # reset the counter
                    target_net_counter = 0

            # make a small simulation to get the rewards of the net for an
            # episode
            if exists(reward_curve_steps_per_point):
                overall_return, last_reward, _, _, _ = self.greedy_simulation(
                    q_net=q_net,
                    environment=copy.deepcopy(environment),
                    start_state=episode_start_state,
                    steps=reward_curve_steps_per_point,
                    device=device,
                )
                for idx, reward_curve_mode_ in enumerate(reward_curve_mode):
                    if reward_curve_mode_ == "return":
                        step_reward = overall_return
                    elif reward_curve_mode_ == "last_reward":
                        step_reward = last_reward
                    elif reward_curve_mode_ == "best_reward":
                        step_reward = training_best_reward
                    elif reward_curve_mode_ == "best_htc":
                        step_reward = training_best_htc

                    # update learning curves
                    reward_curves[idx].update(
                        step_reward, step, learning_rate=lr, epsilon=epsilon
                    )

            # reduction of epsilon at each episode, with a min value of min_eps
            epsilon -= reduce_eps
            epsilon = max(epsilon, min_eps)

            # reduction of learning rate at each episode, with a min value of
            # min_lr
            lr -= lr * reduce_perc_lr / 100
            lr = max(lr, min_lr)
            # update optimizer with next learning rate
            for g in optimizer.param_groups:
                g["lr"] = lr

            # update the step number
            step += 1

            # output convergence info if requested
            if self.verbose and step % debug_counter == 0:
                logging.debug(f"The loss in step {step} has been {loss:.3}")

        # -------------------------------------------------------------------------
        # Step 2: save relevant training products
        # -------------------------------------------------------------------------
        self.q_net = q_net
        self.visited_states = visited_states
        self.known_states_list = known_states_list

        self.loss_curve = loss_curve
        self.max_htc_vs_known_state = max_htc_vs_known_state
        if exists(reward_curve_steps_per_point):
            for idx, reward_curve_mode_ in enumerate(reward_curve_mode):
                if reward_curve_mode_ == "return":
                    self.reward_curve_return = reward_curves[idx]
                elif reward_curve_mode_ == "last_reward":
                    self.reward_curve_last_reward = reward_curves[idx]
                elif reward_curve_mode_ == "best_reward":
                    self.reward_curve_best_reward = reward_curves[idx]
                elif reward_curve_mode_ == "best_htc":
                    self.reward_curve_best_htc = reward_curves[idx]

        if save_q_net:  # [1]
            torch.save(
                q_net.state_dict(),
                f"./data/{self.save_folder}/q_net_{in_dim}_inputs.pt",
            )

        # -------------------------------------------------------------------------
        # Step 3: plot relevant data and save their figures and objects
        # -------------------------------------------------------------------------
        if plot_learning_curves:
            # obtain units label
            units_label = HTC_units_label(environment.reward_reduction_factor)

            loss_curve.plot(
                title="",
                xlabel="Training step",
                ylabel="MAE loss",
                plot_epsilon=True,
                plot_lr=False,
                save_path=f"./img/{self.save_folder}/Loss_learning_curve_{environment.n_layers}.png",
            )

            max_htc_vs_known_state.plot(
                title="",
                xlabel="Found states",
                ylabel="Largest HTC ($\mathdefault{W/m^2K}$)",
                plot_epsilon=False,
                plot_lr=False,
                save_path=f"./img/{self.save_folder}/Max_htc_per_known_state_{environment.n_layers}.png",
            )

            if exists(reward_curve_steps_per_point):
                for idx, reward_curve_mode_ in enumerate(reward_curve_mode):
                    if reward_curve_mode_ == "return":
                        reward_ylabel = f"Return ({units_label})"
                        sufix = "return"
                    elif reward_curve_mode_ == "last_reward":
                        reward_ylabel = (
                            f"Last reward of greedy simulation ({units_label})"
                        )
                        sufix = "last_reward"
                    elif reward_curve_mode_ == "best_reward":
                        reward_ylabel = f"Maximum reward ({units_label})"
                        sufix = "best_reward"
                    elif reward_curve_mode_ == "best_htc":
                        reward_ylabel = (
                            f"Maximum heat transfer coefficient ({units_label})"
                        )
                        sufix = "best_htc"

                    reward_curves[idx].plot(
                        title="",
                        xlabel="Training step",
                        ylabel=reward_ylabel,
                        plot_epsilon=True,
                        plot_lr=False,
                        y_divisor=(
                            environment.reward_reduction_factor
                            if reward_curve_mode_ == "best_htc"
                            else None
                        ),
                        save_path=f"./img/{self.save_folder}/Reward_learning_curve_{environment.n_layers}_{sufix}.png",
                    )
