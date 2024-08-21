import datetime
import numpy as np
import torch as T


def str2timedelta(dmhs: str) -> datetime.timedelta:
    """Convert a string in the format "d:h:m:s" to a timedelta object."""
    d, h, m, s = map(int, dmhs.split(":"))
    return datetime.timedelta(days=d, hours=h, minutes=m, seconds=s)


def timedelta2str(td: datetime.timedelta) -> str:
    """Convert a timedelta object to a string in the format "d:h:m:s"."""
    d = td.days
    h = td.seconds // 3600
    m = (td.seconds // 60) % 60
    s = td.seconds % 60
    return f"{d:02d}:{h:02d}:{m:02d}:{s:02d}"


def no_inf(x: np.ndarray | T.Tensor, skip_nan=False) -> np.ndarray | T.Tensor:
    """
    Removes the infinities from an array.
    Also removes the NaNs by default.
    """
    if skip_nan:
        if isinstance(x, np.ndarray):
            return x[~np.isinf(x)]
        elif isinstance(x, T.Tensor):
            return x[~T.isinf(x)]
        else:
            raise ValueError("x must be a numpy array or a torch tensor.")
    else:
        return no_inf(no_nan(x), skip_nan=True)


def no_nan(x: np.ndarray | T.Tensor) -> np.ndarray | T.Tensor:
    """Removes the NaNs from an array."""
    if isinstance(x, np.ndarray):
        return x[~np.isnan(x)]
    elif isinstance(x, T.Tensor):
        return x[~T.isnan(x)]
    else:
        raise ValueError("x must be a numpy array or a torch tensor.")


def try_or_negative(f, *args, **kwargs):
    """Evaluate a function and return -1 if an exception is raised."""
    try:
        return f(*args, **kwargs)
    except:
        return -1
