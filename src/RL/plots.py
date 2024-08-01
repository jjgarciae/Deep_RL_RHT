import logging
import os
import warnings
from collections import Iterable
from numbers import Number
from pathlib import Path
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import axes, figure

from src.RL_simulations.global_vars import PLOT_FONT
from src.utils import InputError, exists

# set matplotlib font
plt.rcParams["font.family"] = PLOT_FONT


def save_fig_df(
    fig_save_path: str,
    df: Optional[pd.DataFrame] = None,
    x: Optional[np.typing.ArrayLike] = None,
    y: Optional[np.typing.ArrayLike] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs,
) -> None:
    """Retrieve and save data employed for figure generation of functions below.
    Store it at `fig_save_path` but with `.parquet` extension.

    Warnings
    --------
    Considered **kwargs will be stored with their name as variables of the saved
    dataframe.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/3548673/how-can-i-replace-or-strip-an-extension-from-a-filename-in-python
    .. [2] https://stackoverflow.com/questions/23177439/how-to-check-if-a-dictionary-is-empty
    """
    # check df or x/y are given, not both
    if exists(x) and exists(y) and exists(df):
        raise InputError("Only `df` or `x` and `y` must be provided.")

    if exists(x) and exists(y):
        # save figure data
        dict_fig = {xlabel: x, ylabel: y}
        df_fig = pd.DataFrame(dict_fig)
    elif exists(df):
        df_fig = df
    else:
        raise InputError(
            "Not enough input data. Either `df` or `x` and `y` must be provided."
        )

    # save additional inputs
    if bool(kwargs):  # [2]
        for key, value in kwargs.items():
            if value is not None and value.size != 0:
                df_fig[key] = value

    # save the df
    df_fig.to_parquet(os.path.splitext(fig_save_path)[0] + ".parquet")  # [1]


class learning_curve:
    """Generic learning curve.

    Plot a quatification of how good/bad is performing a model vs the number of
    training iterations of the model. It shows the progress of the model along
    it training.

    Classic learning curves plots the model's performance for training/validation
    sets along the number of processed training samples.
    """

    def __init__(self):
        self.performance = np.array([])
        self.iterations = np.array([])
        self.learning_rate = np.array([])
        self.epsilon = np.array([])

    def update(
        self,
        performance: float,
        iteration: int,
        learning_rate: Optional[float] = None,
        epsilon: Optional[float] = None,
    ):
        """Update the final arrays to plot.

        Parameters
        ----------
        performance : float
            Performance of the model in an specific iteration.
        iteration : int
            Current iteration number.
        learning_rate : float, optional
            Current learning rate value.
            If None, it will be not represented.
        epsilon : float, optional
            Current epsilon value.
            If None, it will be not represented.
        """
        self.performance = np.append(self.performance, performance)
        self.iterations = np.append(self.iterations, iteration)

        if exists(learning_rate):
            self.learning_rate = np.append(self.learning_rate, learning_rate)

        if exists(epsilon):
            self.epsilon = np.append(self.epsilon, epsilon)

    def plot(
        self,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        save_path: Optional[str] = None,
        hline: Optional[float] = None,
        vline: Optional[float] = None,
        plot_epsilon: bool = True,
        plot_lr: bool = True,
        x_divisor: Optional[float] = None,
        y_divisor: Optional[float] = None,
        xlim: Optional[list[float]] = None,
        ylim: Optional[list[float]] = None,
        labels_fontsize: int = 20,
        ticks_fontsize: int = 18,
    ) -> tuple[figure.Figure, axes.Axes, axes.Axes, np.ndarray, np.ndarray]:
        """Plot the learning curve with stored values.

        Parameters
        ----------
        title: str, optional
            Figure title. By defaul, "".
        xlabel: str, optional
            Figure x-label. By defaul, "".
        ylabel: str, optional
            Figure main y-label. By default, "".
        save_path: Optional[str], optional
            Figure save path [TEMPORAL] & figure data save path. If None, do not
            save it. By default, None.
        hline : Optional[float], optional
            Plot an horizontal dashed line at selected value.
            Value will be selected with respect the principal y-axis.
            If None, skip this line.
            By default, None.
        vline : Optional[float], optional
            Plot an vertical dashed line at selected value.
            Value will be selected with respect the principal x-axis.
            If None, skip this line.
            By default, None.
        plot_epsilon: bool, optional
            Indicates if plot epsilon curve. By default, True.
        plot_lr: bool, optional
            Indicates if plot learning rate curve. By default, True.
        x_divisor : Optional[float], optional
            Number by which DIVIDE `x` values. Useful to normalize x values.
            If None, do not change `x` values. By default, None.
        y_divisor : Optional[float], optional
            Number by which DIVIDE `y` values. Useful to normalize y values.
            If None, do not change `y` values. By default, None.
        xlim: Optional[list[float]], optional
            Set the limits of the x-axis. If None, let matplotlib set it
            automatically. By defult, None.
        ylim: Optional[list[float]], optional
            Set the limits of the y-axis. If None, let matplotlib set it
            automatically. By defult, None.
        labels_fontsize: int, optional
            Fontsize of labels and legend. Title 4 points plus. By default, 20.
        ticks_fontsize: int, optional
            Fontsize of axis ticks. By default, 18.

        References
        ----------
        .. [1] https://stackoverflow.com/questions/14762181/adding-a-y-axis-label-to-secondary-y-axis-in-matplotlib
        .. [2] https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend
        .. [3] https://stackoverflow.com/questions/35020409/how-to-rotate-secondary-y-axis-label-so-it-doesnt-overlap-with-y-ticks-matplot
        .. [4] https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.tick_params.html
        """
        if plot_epsilon and self.epsilon.shape[0] == 0:
            plot_epsilon = False
            warnings.warn("Epsilon curve selected but no epsilon data found. Skipped.")
        if plot_lr and self.learning_rate.shape[0] == 0:
            plot_lr = False
            warnings.warn(
                "Learning rate curve selected but no learning rate data found. Skipped."
            )

        # divide x/y values by selected number
        x_divisor = 1 if x_divisor is None else x_divisor
        y_divisor = 1 if y_divisor is None else y_divisor

        iterations = self.iterations / x_divisor
        performance = self.performance / y_divisor

        # plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.tick_params(
            direction="in", top=True, right=True, labelsize=ticks_fontsize
        )  # [4]
        plot_lines = []

        # create secondary axis only if it is necessary
        ax2 = None
        if plot_lr or plot_epsilon:
            ax2 = ax.twinx()
            ax2.tick_params(direction="in", labelsize=ticks_fontsize)  # [4]

        # learning curve
        plot_lines += ax.plot(iterations, performance, label="learning curve")

        # learning rate and epsilon curves and y-label [1]
        if plot_lr:
            plot_lines += ax2.plot(self.learning_rate, "r--", label="lr", alpha=0.5)
            ylabel2 = "Learning rate"
        if plot_epsilon:
            plot_lines += ax2.plot(self.epsilon, "k--", label="epsilon", alpha=0.5)
            ylabel2 = "Epsilon"
        if plot_lr and plot_epsilon:
            ylabel2 = "Epsilon and learning rate"

        # add optional horizontal/vertical lines
        if exists(hline):
            ax.axhline(hline, color="g", linestyle=":")
        if exists(vline):
            ax.axvline(vline, color="g", linestyle=":")

        # set plot parameters [1], [2]
        plt.title(title, fontsize=(labels_fontsize + 4))
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel(xlabel, fontsize=labels_fontsize)
        ax.set_ylabel(ylabel, fontsize=labels_fontsize)
        if exists(ax2):
            ax2.set_ylabel(
                ylabel2, fontsize=labels_fontsize, rotation=270, va="bottom"
            )  # [3]
        labels = [line.get_label() for line in plot_lines]
        ax.legend(plot_lines, labels, fontsize=labels_fontsize)

        # if selected, save the created figure
        if exists(save_path):
            fig.savefig(save_path, bbox_inches="tight", dpi=800)
            save_fig_df(
                save_path,
                x=iterations,
                y=performance,
                xlabel=xlabel,
                ylabel=ylabel,
                epsilon=self.epsilon if plot_epsilon else None,
                lr=self.learning_rate if plot_lr else None,
            )

        return fig, ax, ax2, iterations, performance


class mean_learning_curve(learning_curve):
    """Generate a mean learning curve from several learning curve objects."""

    def __init__(
        self, learning_curves: np.ndarray, xrange: Literal["min", "max"] = "min"
    ):
        """Create mean learning curves from several learning curves.

        Parameters
        ----------
        learning_curves : np.ndarray[learning_curve]
            Set of learning curves to compute the mean.
        xrange : Literal["min", "max"]
            Select the x axis range.
                * min : extent x axis until the number of episodes of the
                  shortest simulation.
                * max : extent x axis until the number of episodes of the
                  largest simulation.
            By default, "min".

        Warnings
        --------
        We supose the `learning_rate` and `epsilon` attributes have the same
        decaying rates for all learning curves, so we plot the values of the
        longest simulation.

        References
        ----------
        .. [1] https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html
        .. [2] https://numpy.org/doc/stable/reference/generated/numpy.nanstd.html#numpy.nanstd
        """
        # check number of episodes for each curve and the max and min number of episodes
        n_episodes_curves = np.array(
            [len(learning_curve.iterations) for learning_curve in learning_curves]
        )
        max_n_episodes = n_episodes_curves.max()
        min_n_episodes = n_episodes_curves.min()
        max_episod_diff = max_n_episodes - min_n_episodes

        # initialize some parameters
        one_learning_curve = learning_curves[0]
        curves_performance = []

        # check if all learning curves have the same number of episodes
        if not np.array_equal(
            n_episodes_curves, np.repeat(max_n_episodes, len(n_episodes_curves))
        ):
            logging.info(
                "Number of episodes are different between learning curves. "
                f"Maximum difference is {max_episod_diff} episodes."
            )

            for learn_curve in learning_curves:
                # if not and xrange is "max", set left number of elements as np.nan
                # for the performance
                if xrange == "max":
                    curve_diff = max_n_episodes - len(learn_curve.iterations)
                    # add nans to performance (iterations will be the max)
                    curves_performance.append(
                        np.append(
                            learn_curve.performance, np.repeat(np.nan, curve_diff)
                        )
                    )
                # if not and xrange is "min", crop excess number of elements of the
                # performance
                elif xrange == "min":
                    curve_diff = len(learn_curve.iterations) - min_n_episodes
                    # crop performance (iterations will be the min)
                    curves_performance.append(learn_curve.performance[0:min_n_episodes])

                # get iterations data
                if curve_diff == 0:
                    self.iterations = learn_curve.iterations

                    # DISCLAIMER: we supose the learning_rate and epsilon
                    # parameters have the same decaying rates for all
                    # learning curves, so we plot the values of the shortest
                    # simulation
                    self.learning_rate = learn_curve.learning_rate
                    self.epsilon = learn_curve.epsilon
        else:
            curves_performance = [
                learn_curve.performance for learn_curve in learning_curves
            ]
            self.iterations = one_learning_curve.iterations

            # DISCLAIMER: we supose the learning_rate and epsilon parameters
            # have the same decaying rates for all learning curves
            self.learning_rate = one_learning_curve.learning_rate
            self.epsilon = one_learning_curve.epsilon

        # get performance and its standard deviation
        self.performance = np.nanmean(curves_performance, axis=0)  # [1]
        self.std = np.nanstd(curves_performance, axis=0)  # [2]

    def update(self, **args):
        warnings.warn(
            "`mean_learning_curve.update` method does not do anything. "
            "Input curves must be already created."
        )

    def plot(
        self,
        plot_std: bool = True,
        **kwargs,
    ) -> tuple[figure.Figure, axes.Axes, axes.Axes]:
        """Plot the learning curve with stored values.

        Parameters
        ----------
        plot_std : bool, optional
            Choose to plot the standard devidation of the mean curve.
            By default, True.

        See Also
        --------
        learning_curve.plot : where input kwargs are defined.

        References
        ----------
        .. [1] https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.fill_between.html#matplotlib.axes.Axes.fill_between
        """
        fig, axs, axs2, iterations, performance = super().plot(**kwargs)

        # divide remainding std values by selected number
        y_divisor = kwargs.get("y_divisor", None)
        y_divisor = 1 if y_divisor is None else y_divisor

        curve_std = self.std / y_divisor

        # add optional standard deviation
        if plot_std:
            upper_std = performance + curve_std
            lower_std = performance - curve_std

            axs.fill_between(iterations, lower_std, upper_std, alpha=0.2)  # [1]

        # if selected, save again the figure, adding standard deviation
        save_path = kwargs.get("save_path", None)
        if exists(save_path):
            fig.savefig(save_path, bbox_inches="tight", dpi=800)
            save_fig_df(
                save_path,
                x=iterations,
                y=performance,
                xlabel=kwargs.get("xlabel", None),
                ylabel=kwargs.get("ylabel", None),
                y_std=curve_std,
                epsilon=self.epsilon,
                lr=self.learning_rate,
            )

        return fig, axs, axs2


class merge_learning_curves_figs(learning_curve):
    def __init__(
        self,
        learning_curves: Iterable,
        curve_fill: Optional[Union[Literal["last"], Number]] = None,
    ):
        """Create a figure with input learning curves.

        Parameters
        ----------
        learning_curves : Iterable[learning_curve]
            Set of learning curves to plot.
        curve_fill : Union[Literal["last"], Number], optional
            Select the value to assign to y axis for shorter learning curves.
            If "last", use last y value.
            If None, crop longer learning curves to adjust to shorter curves.
            By defaul, None.

        Warnings
        --------
        We supose `epsilon` attributes have the same decaying rates for all
        learning curves, so we plot the values of an arbitrary simulation.
        """
        # check number of iterations for each curve and the max and min number
        n_iterations = np.array(
            [len(learning_curve.iterations) for learning_curve in learning_curves]
        )
        max_n_iterations = n_iterations.max()
        min_n_iterations = n_iterations.min()
        max_iter_diff = max_n_iterations - min_n_iterations

        # if std attribute not in learning curves, set it to 0 for code
        # homogeneization
        for idx, learn_curve in enumerate(learning_curves):
            if "std" not in dir(learn_curve):
                learn_curve.std = np.repeat(0, n_iterations[idx])

        # check if all learning curves have the same number of iterations
        # if not, set left number of elements of the performance as `performance_fill`
        if not np.array_equal(
            n_iterations, np.repeat(max_n_iterations, len(n_iterations))
        ):
            logging.info(
                "Number of iterations are different between learning curves. "
                f"Maximum difference is {max_iter_diff} iterations."
            )

            # if we want to fill shorter learning curve
            if exists(curve_fill):
                self.curves_performance = []
                self.curves_std = []
                for learn_curve in learning_curves:
                    curve_diff = max_n_iterations - len(learn_curve.iterations)

                    # add `performance_fill` to performance (iterations are set to
                    # the max)
                    if curve_fill == "last":
                        performance_fill = learn_curve.performance[-1]
                        std_fill = learn_curve.std[-1]

                    self.curves_performance.append(
                        np.append(
                            learn_curve.performance,
                            np.repeat(performance_fill, curve_diff),
                        )
                    )
                    self.curves_std.append(
                        np.append(learn_curve.std, np.repeat(std_fill, curve_diff))
                    )

                    # get iterations data
                    if curve_diff == 0:
                        self.iterations = learn_curve.iterations

            # if we want to crop larger learning curve
            else:
                # just take the shorter number of interactions for y axis
                self.iterations = learning_curves[np.argmin(n_iterations)].iterations

                # clip performance and std to shorter number of n_iterations
                self.curves_performance = [
                    learn_curve.performance[:min_n_iterations]
                    for learn_curve in learning_curves
                ]
                self.curves_std = [
                    learn_curve.std[:min_n_iterations]
                    for learn_curve in learning_curves
                ]

        else:
            self.iterations = learning_curves[0].iterations
            self.curves_performance = [
                learn_curve.performance for learn_curve in learning_curves
            ]
            self.curves_std = [learn_curve.std for learn_curve in learning_curves]

        # DISCLAIMER: we supose epsilon parameter have the same
        # decaying rate for all learning curves, so we plot the
        # values of an arbitrary simulation
        self.epsilon = learning_curves[0].epsilon

    def update(self, **args):
        warnings.warn(
            "`mean_learning_curve.update` method does not do anything. "
            "Input curves must be already created."
        )

    def plot(
        self,
        title: str = "",
        labels: Optional[list[str]] = None,
        xlabel: str = "",
        ylabel: str = "",
        save_path: Optional[str] = None,
        std: bool = True,
        hline: Optional[float] = None,
        vline: Optional[float] = None,
        x_divisor: Optional[float] = None,
        y_divisor: Optional[float] = None,
        xlim: Optional[list[float]] = None,
        ylim: Optional[list[float]] = None,
        labels_fontsize: int = 10,
        ticks_fontsize: int = 8,
    ) -> tuple[figure.Figure, axes.Axes]:
        """Plot the learning curve with stored values.

        See Also
        --------
        * learning_curve.plot : input parameters are described in this method.
        * mean_learning_curve.plot : `std` input parameter is described in this
        method.

        Notes
        -----
        Notice the absence of `plot_epsilon` and `plot_lr` inputs with respect
        `learning_curve.plot` method.
        The reason for that is `merge_learning_curves_figs` class is intendended
        to plot different algorithm curves, which won't necessairly have the
        same epsilon and lr behaviour, or even rely on these hiperparameters.

        References
        ----------
        .. [1] https://stackoverflow.com/questions/678236/how-do-i-get-the-filename-without-the-extension-from-a-path-in-python
        .. [2] https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.tick_params.html
        """
        # plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.tick_params(
            direction="in", top=True, right=True, labelsize=ticks_fontsize
        )  # [2]

        # divide x values by selected number
        x_divisor = 1 if x_divisor is None else x_divisor
        iterations = self.iterations / x_divisor

        # plot each one of the learning curves with its corresponding label
        for idx, performance in enumerate(self.curves_performance):
            if labels is None:
                label = f"learning curve {idx}"
            else:
                label = labels[idx]

            # divide y/std values by selected number
            y_divisor = 1 if y_divisor is None else y_divisor

            performance = performance / y_divisor
            curve_std = self.curves_std[idx] / y_divisor

            if std:
                upper_std = performance + curve_std
                lower_std = performance - curve_std
                ax.fill_between(iterations, lower_std, upper_std, alpha=0.2)

            ax.plot(iterations, performance, label=label)

            # if selected, save the data of the figure
            if exists(save_path):
                filename = Path(save_path).stem  # [1]
                curve_path = save_path.replace(filename, f"{filename}_{label}")
                save_fig_df(
                    curve_path,
                    x=iterations,
                    y=performance,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    y_std=curve_std,
                )

        # add optional horizontal/vertical lines
        if exists(hline):
            ax.axhline(hline, color="g", linestyle=":")
        if exists(vline):
            ax.axvline(vline, color="g", linestyle=":")

        # set plot parameters
        plt.title(title, fontsize=(labels_fontsize + 4))
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel(xlabel, fontsize=labels_fontsize)
        ax.set_ylabel(ylabel, fontsize=labels_fontsize)
        ax.legend(fontsize=labels_fontsize)

        # if selected, save the created figure
        if exists(save_path):
            fig.savefig(save_path, bbox_inches="tight", dpi=800)

        return fig, ax
