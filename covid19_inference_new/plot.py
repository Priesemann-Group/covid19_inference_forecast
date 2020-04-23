# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-04-20 18:50:13
# @Last Modified: 2020-04-21 16:47:16
# ------------------------------------------------------------------------------ #
# Callable in your scripts as e.g. `cov.plot.timeseries()`
# Plot functions and helper classes
# Design ideas:
# * Global Parameter Object?
#   - Maybe only for defaults of function parameters but
#   - Have functions be solid stand-alone and only set kwargs from "rc_params"
# * keep model, trace, ax as the three first arguments for every function
# ------------------------------------------------------------------------------ #

import logging
import datetime
import locale
import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------ #
# Plotting functions
# ------------------------------------------------------------------------------ #


def timeseries(
    model,
    trace,
    varname,
    ax=None,
    add_more_later=False,
    draw_ci_95=rcParams["draw_ci_95"],
    draw_ci_75=rcParams["draw_ci_75"],
    draw_inset=False,
    color_futu=rcParams["color_plot"],
    color_past=rcParams["color_plot"],
    num_days_futu=None,
    **kwargs,
):
    """
        Plot varname into provided ax.
    """


    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()



    return ax

# ------------------------------------------------------------------------------ #
# Parameters, we have to do this first so we can have default arguments
# ------------------------------------------------------------------------------ #


def get_rcparams_default():
    """
        Get a Param (dict) of the default parameters.
        Here we set our default values. Assigned once to module variable
        `rcParamsDefault` on load.
    """
    par = Param(
        locale="en_US",
        date_format="%b %-d",
        date_show_minor_ticks=True,
        rasterization_zorder=-1,
        draw_ci_95=True,
        draw_ci_75=False,
        color_plot="tab:green",
        color_annot="#646464",
    )

    return par


def set_rcparams(par):
    """
        Set the rcparameters used for plotting. provided instance of `Param` has to have the following keys (attributes).

        Attributes
        ----------
        locale : str
            region settings, passed to `setlocale()`. Default: "en_US"

        date_format : str
            Format the date on the x axis of time-like data (see https://strftime.org/)
            example April 1 2020:
            "%m/%d" 04/01, "%-d. %B" 1. April
            Default "%b %-d", becomes April 1

        date_show_minor_ticks : bool
            whether to show the minor ticks (for every day). Default: True

        rasterization_zorder : int or None
            Rasterizes plotted content below this value, set to None to keep everything
            a vector, Default: -1

        draw_ci_95 : bool
            For timeseries plots, indicate 95% Confidence interval via fill between.
            Default: True

        draw_ci_75 : bool,
            For timeseries plots, indicate 75% Confidence interval via fill between.
            Default: False

        color_plot : str,
            Base color used for plots, mpl compatible color code "C0", "#303030"
            Defalt : "tab:green"

        color_annot : str,
            Color to use for annotations
            Default : "#646464"

        Example
        ------
        ```
        pars = cov.plot.get_rcparams_default()
        pars["locale"]="de_DE"
        cov.plot.set_rcparams(pars)
        ```
    """
    for key in get_rcparams_default().keys():
        assert key in par.keys(), "Provide all keys that are in .get_rcparams_default()"

    global rcParams
    rcParams = copy.deepcopy(par)


class Param(dict):
    """
        Paramters Base Class (a tweaked dict)

        We inherit from dict and also provide keys as attributes, mapped to `.get()` of
        dict. This avoids the KeyError: if getting parameters via `.the_parname`, we
        return None when the param does not exist.

        Avoid using keys that have the same name as class functions etc.

        Example
        -------
        ```
        foo = Param(lorem="ipsum")
        print(foo.lorem)
        >>> 'ipsum'
        print(foo.does_not_exist is None)
        >>> True
        ```
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return Param(copy.deepcopy(dict(self), memo=memo))

    @property
    def varnames(self):
        return [*self]


# ------------------------------------------------------------------------------ #
# Formatting helpers
# ------------------------------------------------------------------------------ #

# format yaxis 10_000 as 10 k
format_k = lambda num, _: "${:.0f}\,$k".format(num / 1_000)

# format xaxis, ticks and labels
def format_date_xticks(ax, minor=None):
    locale.setlocale(locale.LC_ALL, rcParams.locale)
    ax.xaxis.set_major_locator(
        matplotlib.dates.WeekdayLocator(interval=1, byweekday=matplotlib.dates.SU)
    )
    if minor is None:
        minor = date_show_minor_ticks
    if minor:
        ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(date_format))


def truncate_number(number, precision):
    return "{{:.{}f}}".format(precision).format(number)


def print_median_CI(arr, prec=2):
    f_trunc = lambda n: truncate_number(n, prec)
    med = f_trunc(np.median(arr))
    perc1, perc2 = (
        f_trunc(np.percentile(arr, q=2.5)),
        f_trunc(np.percentile(arr, q=97.5)),
    )
    return "Median: {}\nCI: [{}, {}]".format(med, perc1, perc2)


def conv_time_to_mpl_dates(arr):
    try:
        return matplotlib.dates.date2num(
            [datetime.timedelta(days=float(date)) + date_begin_sim for date in arr]
        )
    except:
        return matplotlib.dates.date2num(
            datetime.timedelta(days=float(arr)) + date_begin_sim
        )


# ------------------------------------------------------------------------------ #
# init
# ------------------------------------------------------------------------------ #
# set global parameter variables
rcParams = get_rcparams_default()
