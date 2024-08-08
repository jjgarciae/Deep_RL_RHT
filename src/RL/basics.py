import random
from abc import ABC, abstractmethod
from collections.abc import Iterable
from numbers import Number
from pathlib import Path
from typing import Literal, Optional, Union

from src.utils import exists, load_conf


class environment(ABC):
    """Basic environment class."""

    def __init__(
        self, cfg_path: Path = Path("./src/config.toml"), seed: Optional[int] = None
    ):
        """Define basic attributes of environments.

        Attributes
        ----------
        state_space : Iterable[str, Iterable, Number]
            Collection of all the possible states of the problem.
        start_state : Union[str, Iterable, Number]
            Start state of the environment.

        terminal_state : Optional[Union[str, Iterable, Number]], optional
            Terminal state for an episode of the environment, if necessary.
        n_episode_steps : Optional[int], optional
            Number of steps which defines an episode, if necessary.
        episode_count : int, optional
            Count of the steps taken in the episode, if necessary.

        reward_def : Number
            Reward definition of the environment.
        """
        self.cfg = load_conf(cfg_path)
        # fix seed if indicated
        if exists(seed):
            random.seed(seed)

        self.state_space = None
        self.start_state = None

        self.terminal_state = None
        self.n_episode_steps = None

        # initialize episode count
        self.episode_count = 0

        self.reward_def = None

    @abstractmethod
    def step(
        self, state: Union[str, Iterable, Number], action: Union[str, Number]
    ) -> tuple[Union[str, Iterable, Number], Number, bool]:
        """Returns the next state and reward given an input state and action.
        Aditionally, output if the episode is finished if we have an episodic
        task.

        Parameters
        ----------
        state : Union[str, Iterable, Number]
            Current state of the agent.
        action : Union[Literal, Number]
            Action taken by the agent.

        Returns
        -------
        next_state: Union[str, Iterable, Number]
            Next state of the agent
        reward : Number
            Reward due to the action from input state.
        end_episode : bool
            Aditional output if we have an episodic task.
            If True, an episode has ended.
        """
        pass

    @abstractmethod
    def reset(self) -> Union[str, Iterable, Number]:
        """Reset the environment to its basal state:
        * Return initial state
        * Episode counter is reset to zero

        Returns
        -------
        str : Union[str, Iterable, Number]
            Initial state.
        """
        pass


class agent(ABC):
    """Basic agent class."""

    def __init__(
        self,
        algorithm: str,
        actions: Iterable[str, Number],
        verbose: bool = True,
        **kwargs
    ):
        """Define basic attributes of the agent.

        Parameters
        ----------
        algorithm : str
            Determine the RL algorithm to use.
        actions : Iterable[str, Number]
            Set of all posible actions the agent can make in the problem of
            study.
        verbose : bool, optional
            Output info along the code. By default, True.

        **kwargs :
            init_policy : pd.DataFrame
                Initial policy to start.
            init_state_values : pd.DataFrame
                Storage of state values for all states.
            init_action_values : pd.DataFrame
                Storage of action values for all states and actions, i.e.,
                Q-table.
        """
        self.verbose = verbose
        self.algorithm = algorithm
        self.actions = actions

    @abstractmethod
    def _act(
        self,
        mode: Literal["greedy", "epsilon_greedy"],
        state: Union[str, Iterable, Number],
        **kwargs
    ) -> tuple[Union[str, Number], Number]:
        """Returns the action that the agent takes given an state, i. e., the
        application of the policy.

        Parameters
        ----------
        mode : Literal["greedy", "epsilon_greedy"]
            Mode of acting.
        state : Union[str, Iterable, Number]
            Current state of the agent.
        **kwargs :
                epsilon : float
                    Value of epsilon in epsilon greedy policy. With higher
                    epsilon, more exploratory behaviour of the policy.
                state_values : pd.DataFrame
                    Storage of state values for all states.
                action_values : pd.DataFrame
                    Storage of action values for all states and actions, i.e.,
                    Q-table.
                        * columns: actions
                        * rows(index): states
                q_net : QNN
                    Network for the prediction of all action-state values for a
                    given state.
                device : Literal["cuda", "mps", "cpu"]
                    Used device in DRL for training.

        Returns
        -------
        action : Union[str, Number]
            Action taken by the agent.
        q_value : Number
            Q-value of the selected state-action.
        """
        pass

    @abstractmethod
    def train(
        self,
        environment: environment,
        discount_rate: float,
        lr: float,
        epsilon: float,
        plot_learning_curves: bool = True,
        seed: Optional[int] = None,
        **kwargs
    ):
        """Get the optimal policy and action/state values.

        Parameters
        ----------
        environment : environment
            Environment object of the problem.
        discount_rate : float
            Discount rate factor for Reinforcement Learning algorithm.
        lr : float
            Value of the learning rate.
        epsilon : float, optional
            Value of epsilon for epsilon greedy policies.
        plot_learning_curves : bool, optional
            Select to plot learning curves or not.
            By default, True.
        seed : int, optional
            Seed number. If None, do not fix any seed. By default, None.

        **kwargs :
            Any other of the immense number of parameters to detemine for
            training.
        """
        pass
