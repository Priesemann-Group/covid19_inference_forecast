# ------------------------------------------------------------------------------ #
# Old plotting helpers that are still used by some of Jonas' examples
# moving to plot.py
# ------------------------------------------------------------------------------ #

import logging
import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


def get_all_free_RVs_names(model):
    """
        Returns the names of all free parameters of the model

        Parameters
        ----------
        model: pm.Model instance

        Returns
        -------
        : list
            all variable names
    """
    varnames = [str(x).replace("_log__", "") for x in model.free_RVs]
    return varnames


def get_prior_distribution(model, x, varname):
    """
        Given a model and variable name, get the prior that was used for modeling.

        Parameters
        ----------
        model: pm.Model instance
        x: list or array
        varname: string

        Returns
        -------
        : array
            the prior distribution evaluated at x
    """
    return np.exp(model[varname].distribution.logp(x).eval())


def plot_hist(model, trace, ax, varname, colors=("tab:blue", "tab:orange"), bins=50):
    """
        Plots one histogram of the prior and posterior distribution of the variable varname.

        Parameters
        ----------
        model: pm.Model instance
        trace: trace of the model
        ax: matplotlib.axes instance
        varname: string
        colors: list with 2 colornames
        bins:  number or array
            passed to np.hist

        Returns
        -------
            None
    """
    if len(trace[varname].shape) >= 2:
        print("Dimension of {} larger than one, skipping".format(varname))
        ax.set_visible(False)
        return
    ax.hist(trace[varname], bins=bins, density=True, color=colors[1], label="Posterior")
    limits = ax.get_xlim()
    x = np.linspace(*limits, num=100)
    try:
        ax.plot(
            x,
            get_prior_distribution(model, x, varname),
            label="Prior",
            color=colors[0],
            linewidth=3,
        )
    except:
        pass
    ax.set_xlim(*limits)
    ax.set_ylabel("Density")
    ax.set_xlabel(varname)


def plot_cases(
    trace,
    new_cases_obs,
    date_begin_sim,
    diff_data_sim,
    start_date_plot=None,
    end_date_plot=None,
    ylim=None,
    week_interval=None,
    colors=("tab:blue", "tab:orange"),
    country="Germany",
):
    """
        Plots the new cases, the fit, forecast and lambda_t evolution

        Parameters
        ----------
        trace : trace returned by model
        new_cases_obs : array
        date_begin_sim : datetime.datetime
        diff_data_sim : float
            Difference in days between the begin of the simulation and the data
        start_date_plot : datetime.datetime
        end_date_plot : datetime.datetime
        ylim : float
            the maximal y value to be plotted
        week_interval : int
            the interval in weeks of the y ticks
        colors : list with 2 colornames

        Returns
        -------
        figure, axes
    """

    def conv_time_to_mpl_dates(arr):
        return matplotlib.dates.date2num(
            [datetime.timedelta(days=float(date)) + date_begin_sim for date in arr]
        )

    new_cases_sim = trace["new_cases"]
    len_sim = trace["lambda_t"].shape[1]
    if start_date_plot is None:
        start_date_plot = date_begin_sim + datetime.timedelta(days=diff_data_sim)
    if end_date_plot is None:
        end_date_plot = date_begin_sim + datetime.timedelta(days=len_sim)
    if ylim is None:
        ylim = 1.6 * np.max(new_cases_obs)

    num_days_data = len(new_cases_obs)
    diff_to_0 = num_days_data + diff_data_sim
    date_data_end = date_begin_sim + datetime.timedelta(
        days=diff_data_sim + num_days_data
    )
    num_days_future = (end_date_plot - date_data_end).days
    start_date_mpl, end_date_mpl = matplotlib.dates.date2num(
        [start_date_plot, end_date_plot]
    )

    if week_interval is None:
        week_inter_left = int(np.ceil(num_days_data / 7 / 5))
        week_inter_right = int(np.ceil((end_date_mpl - start_date_mpl) / 7 / 6))
    else:
        week_inter_left = week_interval
        week_inter_right = week_interval

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9, 5),
        gridspec_kw={"height_ratios": [1, 3], "width_ratios": [2, 3]},
    )

    ax = axes[1][0]
    time_arr = np.arange(-len(new_cases_obs), 0)
    mpl_dates = conv_time_to_mpl_dates(time_arr) + diff_data_sim + num_days_data
    ax.plot(
        mpl_dates,
        new_cases_obs,
        "d",
        markersize=6,
        label="Data",
        zorder=5,
        color=colors[0],
    )
    new_cases_past = new_cases_sim[:, :num_days_data]
    percentiles = (
        np.percentile(new_cases_past, q=2.5, axis=0),
        np.percentile(new_cases_past, q=97.5, axis=0),
    )
    ax.plot(
        mpl_dates,
        np.median(new_cases_past, axis=0),
        color=colors[1],
        label="Fit (with 95% CI)",
    )
    ax.fill_between(
        mpl_dates, percentiles[0], percentiles[1], alpha=0.3, color=colors[1]
    )
    ax.set_yscale("log")
    ax.set_ylabel("Number of new cases")
    ax.set_xlabel("Date")
    ax.legend()
    ax.xaxis.set_major_locator(
        matplotlib.dates.WeekdayLocator(
            interval=week_inter_left, byweekday=matplotlib.dates.SU
        )
    )
    ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%d"))
    ax.set_xlim(start_date_mpl)

    ax = axes[1][1]

    time1 = np.arange(-len(new_cases_obs), 0)
    mpl_dates = conv_time_to_mpl_dates(time1) + diff_data_sim + num_days_data
    ax.plot(
        mpl_dates,
        new_cases_obs,
        "d",
        label="Data",
        markersize=4,
        color=colors[0],
        zorder=5,
    )

    new_cases_past = new_cases_sim[:, :num_days_data]
    ax.plot(
        mpl_dates,
        np.median(new_cases_past, axis=0),
        "--",
        color=colors[1],
        linewidth=1.5,
        label="Fit with 95% CI",
    )
    percentiles = (
        np.percentile(new_cases_past, q=2.5, axis=0),
        np.percentile(new_cases_past, q=97.5, axis=0),
    )
    ax.fill_between(
        mpl_dates, percentiles[0], percentiles[1], alpha=0.2, color=colors[1]
    )

    time2 = np.arange(0, num_days_future)
    mpl_dates_fut = conv_time_to_mpl_dates(time2) + diff_data_sim + num_days_data
    cases_future = new_cases_sim[:, num_days_data : num_days_data + num_days_future].T
    median = np.median(cases_future, axis=-1)
    percentiles = (
        np.percentile(cases_future, q=2.5, axis=-1),
        np.percentile(cases_future, q=97.5, axis=-1),
    )
    ax.plot(
        mpl_dates_fut,
        median,
        color=colors[1],
        linewidth=3,
        label="forecast with 75% and 95% CI",
    )
    ax.fill_between(
        mpl_dates_fut, percentiles[0], percentiles[1], alpha=0.1, color=colors[1]
    )
    ax.fill_between(
        mpl_dates_fut,
        np.percentile(cases_future, q=12.5, axis=-1),
        np.percentile(cases_future, q=87.5, axis=-1),
        alpha=0.2,
        color=colors[1],
    )

    ax.set_xlabel("Date")
    ax.set_ylabel(f"New confirmed cases in {country}")
    ax.legend(loc="upper left")
    ax.set_ylim(0, ylim)
    func_format = lambda num, _: "${:.0f}\,$k".format(num / 1_000)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(func_format))
    ax.set_xlim(start_date_mpl, end_date_mpl)
    ax.xaxis.set_major_locator(
        matplotlib.dates.WeekdayLocator(
            interval=week_inter_right, byweekday=matplotlib.dates.SU
        )
    )
    ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%d"))

    ax = axes[0][1]

    time = np.arange(-diff_to_0, -diff_to_0 + len_sim)
    lambda_t = trace["lambda_t"][:, :]
    μ = trace["mu"][:, None]
    mpl_dates = conv_time_to_mpl_dates(time) + diff_data_sim + num_days_data

    ax.plot(mpl_dates, np.median(lambda_t - μ, axis=0), color=colors[1], linewidth=2)
    ax.fill_between(
        mpl_dates,
        np.percentile(lambda_t - μ, q=2.5, axis=0),
        np.percentile(lambda_t - μ, q=97.5, axis=0),
        alpha=0.15,
        color=colors[1],
    )

    ax.set_ylabel("effective\ngrowth rate $\lambda_t^*$")

    # ax.set_ylim(-0.15, 0.45)
    ylims = ax.get_ylim()
    ax.hlines(0, start_date_mpl, end_date_mpl, linestyles=":")
    delay = matplotlib.dates.date2num(date_data_end) - np.percentile(
        trace["delay"], q=75
    )
    ax.vlines(delay, ylims[0], ylims[1], linestyles="-", colors=["tab:red"])
    ax.set_ylim(*ylims)
    ax.text(
        delay + 0.5,
        ylims[1] - 0.04 * np.diff(ylims),
        "unconstrained because\nof reporting delay",
        color="tab:red",
        verticalalignment="top",
    )
    ax.text(
        delay - 0.5,
        ylims[1] - 0.04 * np.diff(ylims),
        "constrained\nby data",
        color="tab:red",
        horizontalalignment="right",
        verticalalignment="top",
    )
    ax.xaxis.set_major_locator(
        matplotlib.dates.WeekdayLocator(
            interval=week_inter_right, byweekday=matplotlib.dates.SU
        )
    )
    ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%d"))
    ax.set_xlim(start_date_mpl, end_date_mpl)

    axes[0][0].set_visible(False)

    plt.subplots_adjust(wspace=0.4, hspace=0.3)

    return fig, axes
