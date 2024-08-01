# libraries imports
import logging
import time
import warnings
from math import floor, log10
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import tomli
from matplotlib import pyplot
from tables.exceptions import HDF5ExtError


# custom exceptions and classes
class InputError(ValueError):
    pass


class WIPError(Exception):
    """Custom exception for unfinished code.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/49224770/default-message-in-custom-exception-python
    """

    def __init__(self, msg="Selected option is `work in progress`.", *args, **kwargs):
        super().__init__(msg, *args, **kwargs)


class Conf(dict):
    """Sub-class of dict that overrides __getitem__ to allow for keys not in
    the original dict, defaulting to None.

    Author
    ------
    Komorebi AI Technologies
    """

    def __init__(self, *args, **kwargs):
        """Update dict with all keys from dict"""
        self.update(*args, **kwargs)
        # Parse the config (specifically, change "None" to None and "int" to int)
        self.update(parse_dict(self))

    def __getitem__(self, key):
        """Get key from dict. If not present, return None and raise warning

        Parameters
        ----------
        key : Hashable
            key to get from original dict

        Returns
        -------
            original value in the dict or None if not present
        """
        if key not in self:
            warnings.warn(f"Key '{key}' not in conf. Defaulting to None")
            val = None
        else:
            val = dict.__getitem__(self, key)
        return val


# helpers
def exists(val):
    return val is not None


def timer(start: time, end: time, label: str = "Execution"):
    """Print elapsed time.

    Parameters
    ----------
    start : time
        Start time.
    end : time
        End time.
    label : str, optional
        Optional label for output message.
        By default, "Execution".

    References
    ----------
    .. [1] https://stackoverflow.com/questions/27779677/how-to-format-elapsed-time-from-seconds-to-hours-minutes-seconds-and-milliseco

    Examples
    --------
    logging.basicConfig(level=logging.INFO)

    start = time.time()
    output = main()
    end = time.time()

    timer(start, end)
    """
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        f"------ {label} time: "
        "{:0>2}:{:0>2}:{:05.2f} ------".format(int(hours), int(minutes), seconds)
    )


def tuple_or_list_to_str(list_tuple_list: list[Union[tuple, list]]) -> list[str]:
    """Store the elements of a list with tuples/lists into a list with strings.

    Parameters
    ----------
    list_tuple_list : list[Union[tuple, list]]
        List of tuples/lists to store into a list of strs.

    Returns
    -------
    list[str]
        Elements of the tuple/list stored in a str.

    See Also
    --------
    str_to_tuple_or_list : function to undo this transformation.
    """
    return list(map(lambda x: " ".join(map(str, x)), list_tuple_list))


def str_to_tuple_or_list(
    string: str,
    to: Literal["tuple", "list"],
    separator: str = " ",
    out_dtype: Literal["int", "float", "str"] = "int",
) -> Union[tuple, list]:
    """Store the elements of a string into a tuple/list transformed into selected dtype.

    Parameters
    ----------
    string : str
        String to store into a tuple/list.
    to : Literal["tuple", "list"]
        Selection of the output container.
    separator : str, optional
        Indicate the separator of the elements of the string.
        By default, " ".
    out_dtype : Literal["int", "float", "str"], optional
        Data type of the output elements.
        By default, "int".

    Returns
    -------
    Union[tuple, list]
        Elements of the str stored into a tuple/list.

    See Also
    --------
    tuple_or_list_to_str : function to undo this transformation.

    Examples
    --------
    >>> string = '0 0'
    >>> str_to_tuple_or_list(string, to="list", split = " ", out_dtype = int)
    [0, 0]
    >>> str_to_tuple_or_list(string, to="tuple", split = " ", out_dtype = float)
    (0.0, 0.0)
    """
    if out_dtype == "int":
        dtype = int
    elif out_dtype == "float":
        dtype = float
    elif out_dtype == "str":
        dtype = str
    else:
        raise InputError("Selected `out_dtype` not available.")

    if to == "tuple":
        out = tuple(np.array(string.split(separator)).astype(dtype))
    elif to == "list":
        out = np.array(string.split(separator)).astype(dtype).tolist()
    return out


def check_empty_idx(data: pd.Series, index: str, empty_value: any = np.nan):
    """Check if an index value is considered empty.

    Parameters
    ----------
    data : pd.Series
        Series where index value to check is stored.
    index : str
        Index for which its emptiness is checked.
    empty_value : any, optional
        Value for which indicated index value would be considered empty.
        By default, np.nan.
    """
    empty_flag = False
    if data.empty:
        empty_flag = True

    elif index not in data.index:
        empty_flag = True

    elif np.isnan(empty_value):
        if np.isnan(data[index]):
            empty_flag = True
    else:
        warnings.warn(
            "`Empty_value` is checked with an equality."
            " If it is not a the way to treat `empty_value` dtype, please"
            " contact <eva.ortizm@estudiante.uam.es>."
        )
        if data[index] == empty_value:
            empty_flag = True

    return empty_flag


def load_hdf_to_series(
    file_path: str,
    empty_value: any = np.nan,
    data_key: Optional[str] = None,
    all_index: Optional[list[str]] = None,
) -> pd.Series:
    """Load hdf storage into an pd.Series. If indicated file/key does not
    exists, create it and retrieve `empty_values` for `all_index`.

    Parameters
    ----------
    file_path : str
        Indicates de path of the storage file.
    empty_value: any, optional
        Value which indicates the value of the file is missing.
        By default, np.nan.
    data_key : str, optional
        Key of the file which indicates the data we want to load. It can be
        omitted (None) if the HDF file contains a single pandas object.
        By default, None.
    all_index : Optional[list[str]], optional
        Indexes of the file. Only necessary if the data has not been previously
        stored with key `data_key` in the file.
        By default, None.

    Returns
    -------
    pd.Series
        Data stored in `file_path`.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/17098654/how-to-reversibly-store-and-load-a-pandas-dataframe-to-from-disk
    .. [2] https://stackoverflow.com/questions/51470574/close-hdf-file-after-to-hdf-using-mode-a
    .. [3] https://github.com/pandas-dev/pandas/issues/4409
    """
    # Look for the storage of `data_key` in `file_path`. If it is not stored, create it later
    # in the code. [1]
    try:
        with pd.HDFStore(file_path, mode="r") as storage:
            value_series = pd.read_hdf(storage, key=data_key)
    except HDF5ExtError as e:
        logging.info(
            "\n\nHDF5ExtError raised. It could be due to a previous unexpected"
            " termination.\nError details:\n"
        )
        raise (e)
    # if key error or file not found, initialize a new series to store
    except KeyError or FileNotFoundError:
        value_series = pd.Series(data=empty_value, index=all_index)

        if data_key is None:
            data_key = "data"
            warnings.warn(
                "[FileNotFoundError/KeyError] New file will be created with `data` key."
            )

        with pd.HDFStore(file_path, mode="w") as storage:
            storage.put(
                key=data_key,
                value=value_series,
                format="table",
            )

    return value_series


def load_or_update_series(
    storage: pd.Series,
    desired_index: str,
    function,
    empty_value: any = np.nan,
    **func_inputs,
):
    """Retrieve the value of the `desired_index` in a pd.series. If this value
    is not stored (value `empty_value`), store it.

    Parameters
    ----------
    storage : pd.Series
        Series object which stores desired values.
    desired_index : str
        Index of the storage where the desired value should be stored.
    function
        Function to get missing values of the storage.
        Only necessary if the storage has an empty value.
    empty_value: any, optional
        Value which indicates that the value of the storage is missing.
        By default, np.nan.
    **func_inputs
        Keyword arguments of func.
        Only necessary if the storage has an empty value.

    Returns
    -------
    Union[int, float, str]
        Value stored at `desired_index` in `storage`.
    pd.Series
        `storage` with updated values (if any).
    """

    # Search if the value correspondig to `desired_index` is stored.
    # If it is stored, retrieve its value. Else, calculate its value with the
    # indicated function and its inputs, and store it.
    if check_empty_idx(storage, desired_index, empty_value=empty_value):
        value = function(**func_inputs)
        storage[desired_index] = value
    else:
        value = storage[desired_index]

    return value, storage


def store_series_to_hdf(
    file_path: str,
    data_to_store: pd.Series,
    data_key: str = "value_series",
):
    """Store `data_to_store` into an hdf file under the key `data_key`.

    Parameters
    ----------
    file_path : str
        Indicates de path of the storage file.
    data_to_store : pd.Series
        Data to store in storage file.
    data_key : str, optional
        Key of the file which indicates where to store the data.
        By default, "value_series".

    References
    ----------
    .. [1] https://github.com/pandas-dev/pandas/issues/4409
    """
    try:
        with pd.HDFStore(file_path, mode="w") as storage:
            storage.put(
                key=data_key,
                value=data_to_store,
                format="table",
            )
        storage.close()
    except HDF5ExtError as e:  # [1]
        logging.info(
            "\n\nHDF5ExtError raised. It could be due to try to store several "
            f"new values in the same file {file_path} for different runs.\n\n"
        )
        raise (e)


def set_logging(level: str = "debug"):
    """Set root logger with specific logging level. Capture warnings by warnings module

    Parameters
    ----------
    level : str, optional
        logging level, by default "debug"
    """

    def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
        return " %s:%s: %s:%s" % (filename, lineno, category.__name__, message)

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    warnings.formatwarning = warning_on_one_line
    logging.captureWarnings(True)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)-4.4s] %(message)s", level=numeric_level
    )

    # A large number of DEBUG warnings appear on the log due to matplotlib
    # https://stackoverflow.com/questions/65728037/matplotlib-debug-turn-off-when-python-debug-is-on-to-debug-rest-of-program
    pyplot.set_loglevel(level="info")

    pil_logger = logging.getLogger("PIL")
    pil_logger.setLevel(logging.INFO)


# --------------- config utils --------------- By Komorebi AI Technologies
def load_conf(path: Union[str, Path], key: str = None) -> Conf:
    """Load TOML config as dict-like

    Parameters
    ----------
    path : str
        Path to TOML config file
    key : str, optional
        Section of the conf file to load

    Returns
    -------
    Conf
        Config dictionary

    Author
    ------
    Komorebi AI Technologies
    """
    with open(path, "rb") as f:
        config = tomli.load(f)
    return Conf(config) if key is None else Conf(config[key])


def parse_str(x: str):
    """Parses a string value x:
    - If x is "none", returns None
    - If x is numeric, returns int(x) or float(x)
    - In any other case, returns the original x

    Author
    ------
    Komorebi AI Technologies
    """
    if not isinstance(x, str):  # Only True when value is not str
        return x
    elif x.lower() == "none":
        return None
    elif x.isnumeric():  # Only True when value is int
        return int(x)
    elif isfloat(x):  # Only True when value is float
        return float(x)
    else:
        return x


def parse_list(ser: list) -> list:
    """Parses the elements of a list

    Author
    ------
    Komorebi AI Technologies
    """
    return [parse_str(x) for x in ser]


def parse_dict(d: dict) -> dict:
    """Parses of the elements of a dictionary

    Author
    ------
    Komorebi AI Technologies
    """
    out_d = d.copy()
    for key, value in d.items():
        if isinstance(value, dict):
            out_d.update({key: parse_dict(value)})
        elif isinstance(value, list):
            out_d.update({key: parse_list(value)})
        else:
            out_d.update({key: parse_str(value)})
    return out_d


def isfloat(value):
    """Returns True when the value can be converted to float

    Author
    ------
    Komorebi AI Technologies
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def find_exp(number) -> int:
    """Obtain the exponent from scientific notation.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/64183806/extracting-the-exponent-from-scientific-notation
    """
    base10 = log10(abs(number))
    return floor(base10)


def scientific_notation_label(number: float) -> str:
    """Return scientific notation label of input number.

    Parameters
    ----------
    number : float
        Number for which get scientific notation.

    Returns
    -------
    str
        String with scientific notation.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/21226868/superscript-in-python-plots
    .. [2] https://matplotlib.org/2.0.2/users/mathtext.html
    """
    exponent = find_exp(number)
    base = number / 10**exponent

    label = (
        f"$10^{int(exponent)}$" if base == 1 else f"${base} \cdot 10^{int(exponent)}$"
    )  # [1], [2]
    return label
