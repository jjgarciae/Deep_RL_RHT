from typing import Optional

from src.utils import scientific_notation_label


def HTC_units_label(reduction_factor: Optional[int] = None):
    """Obtain HTC units label.

    Parameters
    ----------
    reduction_factor : Optional[int], optional
        Reduction factor applied to HTC for printing it at unit level, by
        default None.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/14324477/bold-font-weight-for-latex-axes-label-in-matplotlib
    .. [2] https://stackoverflow.com/questions/72792392/how-to-use-latex-within-an-f-string-expression-in-matplotlib-no-variables-in-eq
    .. [3] https://stackoverflow.com/questions/53679650/how-to-change-font-in-which-is-in-math-form-in-python
    """
    # obtain the exponent form of `reward_reduction_factor`
    if reduction_factor is not None:
        reward_reduction_label = scientific_notation_label(reduction_factor).replace(
            "$", ""
        )
    else:
        reward_reduction_label = ""
    # display previous label and units [3]
    # DEPRECATED: in bold form [1, 2]
    return f"${{\mathdefault{{{reward_reduction_label} W/m^2K}}}}$"
