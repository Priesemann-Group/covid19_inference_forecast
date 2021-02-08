# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-04-20 18:50:13
# @Last Modified: 2020-10-25 21:33:13
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
import re

import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats

log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------ #
# Time series plotting functions
# ------------------------------------------------------------------------------ #


def timeseries_overview(
    model,
    trace,
    start=None,
    end=None,
    region=None,
    color=None,
    save_to=None,
    offset=0,
    annotate_constrained=True,
    annotate_watermark=True,
    axes=None,
    forecast_label="Forecast",
    forecast_heading=r"$\bf Forecasts\!:$",
    add_more_later=False,
):
    """
        Create the time series overview similar to our paper.
        Dehning et al. arXiv:2004.01105
        Contains $\lambda$, new cases, and cumulative cases.

        Parameters
        ----------
        model : model instance

        trace : trace instance
            needed for the data

        offset : int
            offset that needs to be added to the (cumulative sum of) new cases at time
            model.data_begin to arrive at cumulative cases

        start : datetime.datetime
            only used to set xrange in the end
        end : datetime.datetime
            only used to set xrange in the end
        color : string
            main color to use, default from rcParam
        save_to : string or None
            path where to save the figures. default: None, not saving figures
        annotate_constrained : bool
            show the unconstrained constrained annotation in lambda panel
        annotate_watermark : bool
            show our watermark
        axes : np.array of mpl axes
            provide an array of existing axes (from previously calling this function)
            to add more traces. Data will not be added again. Ideally call this first
            with `add_more_later=True`
        forecast_label : string
            legend label for the forecast, default: "Forecast"
        forecast_heading : string
            if `add_more_later`, how to label the forecast section.
            default: "$\bf Forecasts\!:$",
        add_more_later : bool
            set this to true if you plan to add multiple models to the plot. changes the layout (and the color of the fit to past data)

        Returns
        -------
            fig : mpl figure
            axes : np array of mpl axeses (insets not included)

        TODO
        ----
        * Replace `offset` with an instance of data class that should yield the
          cumulative cases. we should not to calculations here.
    """

    figsize = (6, 6)
    # ylim_new = [0, 2_000]
    # ylim_new_inset = [50, 17_000]
    # ylim_cum = [0, 20_000]
    # ylim_cum_inset = [50, 300_000]
    ylim_lam = [-0.15, 0.45]

    label_y_new = f"Daily new\nreported cases"
    label_y_cum = f"Total\nreported cases"
    label_y_lam = f"Effective\ngrowth rate $\lambda^\\ast (t)$"
    label_leg_data = "Data"
    label_leg_dlim = f"Updated on\n{datetime.datetime.now().strftime('%Y/%m/%d')}"

    if rcParams["locale"].lower() == "de_de":
        label_y_new = f"Täglich neu\ngemeldete Fälle"
        label_y_cum = f"Gesamtzahl\ngemeldeter Fälle"
        label_y_lam = f"Effektive\nWachstumsrate"
        label_leg_data = "Daten"
        label_leg_dlim = f"Daten bis\n{model.data_end.strftime('%-d. %B %Y')}"

    letter_kwargs = dict(x=-0.25, y=1, size="x-large")

    # per default we assume no hierarchical
    if region is None:
        region = ...

    axes_provided = False
    if axes is not None:
        log.debug("Provided axes, adding new content")
        axes_provided = True

    color_data = rcParams.color_data
    color_past = rcParams.color_model
    color_fcast = rcParams.color_model
    color_annot = rcParams.color_annot
    if color is not None:
        color_past = color
        color_fcast = color

    if axes_provided:
        fig = axes[0].get_figure()
    else:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=figsize,
            gridspec_kw={"height_ratios": [2, 3, 3]},
            constrained_layout=True,
        )
        if add_more_later:
            color_past = "#646464"

    if start is None:
        start = model.data_begin
    if end is None:
        end = model.sim_end

    # insets are not reimplemented yet
    insets = []
    insets_only_two_ticks = True
    draw_insets = False

    # ------------------------------------------------------------------------------ #
    # lambda*, effective growth rate
    # ------------------------------------------------------------------------------ #
    ax = axes[0]
    mu = trace["mu"][:, None]
    lambda_t, x = _get_array_from_trace_via_date(model, trace, "lambda_t")
    y = lambda_t[:, :, region] - mu
    _timeseries(x=x, y=y, ax=ax, what="model", color=color_fcast)
    ax.set_ylabel(label_y_lam)
    ax.set_ylim(ylim_lam)

    if not axes_provided:
        ax.text(s="A", transform=ax.transAxes, **letter_kwargs)
        ax.hlines(0, x[0], x[-1], linestyles=":")
        if annotate_constrained:
            try:
                # depending on hierchy delay has differnt variable names.
                # get the shortest one. todo: needs to be change depending on region.
                delay_vars = [var for var in trace.varnames if "delay" in var]
                delay_var = delay_vars.sort(key=len)[0]
                delay = mpl.dates.date2num(model.data_end) - np.percentile(
                    trace[delay_var], q=75
                )
                ax.vlines(delay, -10, 10, linestyles="-", colors=color_annot)
                ax.text(
                    delay + 1.5,
                    0.4,
                    "unconstrained due\nto reporting delay",
                    color=color_annot,
                    horizontalalignment="left",
                    verticalalignment="top",
                )
                ax.text(
                    delay - 1.5,
                    0.4,
                    "constrained\nby data",
                    color=color_annot,
                    horizontalalignment="right",
                    verticalalignment="top",
                )
            except Exception as e:
                log.debug(f"{e}")

    # --------------------------------------------------------------------------- #
    # New cases, lin scale first
    # --------------------------------------------------------------------------- #
    ax = axes[1]

    y_past, x_past = _get_array_from_trace_via_date(
        model, trace, "new_cases", model.data_begin, model.data_end
    )
    y_past = y_past[:, :, region]

    y_data = model.new_cases_obs[:, region]

    x_data = pd.date_range(start=model.data_begin, end=model.data_end)

    # data points and annotations, draw only once
    if not axes_provided:
        ax.text(s="B", transform=ax.transAxes, **letter_kwargs)
        _timeseries(
            x=x_data,
            y=y_data,
            ax=ax,
            what="data",
            color=color_data,
            zorder=5,
            label=label_leg_data,
        )
        # model fit
        _timeseries(
            x=x_past, y=y_past, ax=ax, what="model", color=color_past,  # label="Fit",
        )
        if add_more_later:
            # dummy element to separate forecasts
            ax.plot(
                [], [], "-", linewidth=0, label=forecast_heading,
            )

    # model fcast
    y_fcast, x_fcast = _get_array_from_trace_via_date(
        model, trace, "new_cases", model.fcast_begin, model.fcast_end
    )
    y_fcast = y_fcast[:, :, region]
    _timeseries(
        x=x_fcast,
        y=y_fcast,
        ax=ax,
        what="fcast",
        color=color_fcast,
        label=f"{forecast_label}",
    )
    ax.set_ylabel(label_y_new)
    # ax.set_ylim(ylim_new)
    prec = 1.0 / (np.log10(ax.get_ylim()[1]) - 2.5)
    if prec < 2.0 and prec >= 0:
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(_format_k(int(prec)))
        )

    # ------------------------------------------------------------------------------ #
    # total cases, still needs work because its not in the trace, we cant plot it
    # due to the lacking offset from new to cumulative cases, we cannot calculate
    # either.
    # ------------------------------------------------------------------------------ #

    ax = axes[2]

    y_past, x_past = _get_array_from_trace_via_date(
        model, trace, "new_cases", model.data_begin, model.data_end
    )
    y_past = y_past[:, :, region]

    y_data = model.new_cases_obs[:, region]
    x_data = pd.date_range(start=model.data_begin, end=model.data_end)

    x_data, y_data = _new_cases_to_cum_cases(x_data, y_data, "data", offset)
    x_past, y_past = _new_cases_to_cum_cases(x_past, y_past, "trace", offset)

    # data points and annotations, draw only once
    if not axes_provided:
        ax.text(s="C", transform=ax.transAxes, **letter_kwargs)
        _timeseries(
            x=x_data,
            y=y_data,
            ax=ax,
            what="data",
            color=color_data,
            zorder=5,
            label=label_leg_data,
        )
        # model fit
        _timeseries(
            x=x_past, y=y_past, ax=ax, what="model", color=color_past,  # label="Fit",
        )
        if add_more_later:
            # dummy element to separate forecasts
            ax.plot(
                [], [], "-", linewidth=0, label=forecast_heading,
            )

    # model fcast, needs to start one day later, too. use the end date we got before
    y_fcast, x_fcast = _get_array_from_trace_via_date(
        model, trace, "new_cases", model.fcast_begin, model.fcast_end
    )
    y_fcast = y_fcast[:, :, region]

    # offset according to last cumulative model point
    x_fcast, y_fcast = _new_cases_to_cum_cases(
        x_fcast, y_fcast, "trace", y_past[:, -1, None]
    )

    _timeseries(
        x=x_fcast,
        y=y_fcast,
        ax=ax,
        what="fcast",
        color=color_fcast,
        label=f"{forecast_label}",
    )
    ax.set_ylabel(label_y_cum)
    # ax.ylim(ylim_cum)
    prec = 1.0 / (np.log10(ax.get_ylim()[1]) - 2.5)
    if prec < 2.0 and prec >= 0:
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(_format_k(int(prec)))
        )

    # --------------------------------------------------------------------------- #
    # Finalize
    # --------------------------------------------------------------------------- #

    for ax in axes:
        ax.set_rasterization_zorder(rcParams.rasterization_zorder)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlim(start, end)
        _format_date_xticks(ax)

        # biweekly, remove every second element
        if not axes_provided:
            for label in ax.xaxis.get_ticklabels()[1::2]:
                label.set_visible(False)

    for ax in insets:
        ax.set_xlim(start, model.data_end)
        ax.yaxis.tick_right()
        ax.set_yscale("log")
        if insets_only_two_ticks is True:
            format_date_xticks(ax, minor=False)
            for label in ax.xaxis.get_ticklabels()[1:-1]:
                label.set_visible(False)
            print(ax.xticks)
        else:
            format_date_xticks(ax)
            for label in ax.xaxis.get_ticklabels()[1:-1]:
                label.set_visible(False)

    # legend
    leg_loc = "upper left"
    if draw_insets == True:
        leg_loc = "upper right"
    ax = axes[2]
    ax.legend(loc=leg_loc)
    ax.get_legend().get_frame().set_linewidth(0.0)
    ax.get_legend().get_frame().set_facecolor("#F0F0F0")
    # styling legend elements individually does not work. seems like an mpl bug,
    # changes to fontproperties get applied to all legend elements.
    # for tel in ax.get_legend().get_texts():
    #     if tel.get_text() == "Forecasts:":
    #         # tel.set_fontweight("bold")

    if annotate_watermark:
        _add_watermark(axes[1])

    fig.suptitle(
        # using script run time. could use last data point though.
        label_leg_dlim,
        x=0.15,
        y=1.075,
        verticalalignment="top",
        # fontsize="large",
        fontweight="bold",
        # loc="left",
        # horizontalalignment="left",
    )

    # plt.subplots_adjust(wspace=0.4, hspace=0.25)
    if save_to is not None:
        plt.savefig(
            save_to + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.05,
        )
        plt.savefig(
            save_to + ".png", dpi=300, bbox_inches="tight", pad_inches=0.05,
        )

    # add insets to returned axes. maybe not, general axes style would be applied
    # axes = np.append(axes, insets)

    return fig, axes


def _timeseries(
    x,
    y,
    ax=None,
    what="data",
    draw_ci_95=None,
    draw_ci_75=None,
    draw_ci_50=None,
    **kwargs,
):
    """
        low-level function to plot anything that has a date on the x-axis.

        Parameters
        ----------
        x : array of datetime.datetime
            times for the x axis

        y : array, 1d or 2d
            data to plot. if 2d, we plot the CI as fill_between (if CI enabled in rc
            params)
            if 2d, then first dim is realization and second dim is time matching `x`
            if 1d then first tim is time matching `x`

        ax : mpl axes element, optional
            plot into an existing axes element. default: None

        what : str, optional
            what type of data is provided in x. sets the style used for plotting:
            * `data` for data points
            * `fcast` for model forecast (prediction)
            * `model` for model reproduction of data (past)

        kwargs : dict, optional
            directly passed to plotting mpl.

        Returns
        -------
            ax
    """

    # ------------------------------------------------------------------------------ #
    # Default parameter
    # ------------------------------------------------------------------------------ #

    if draw_ci_95 is None:
        draw_ci_95 = rcParams["draw_ci_95"]

    if draw_ci_75 is None:
        draw_ci_75 = rcParams["draw_ci_75"]

    if draw_ci_50 is None:
        draw_ci_50 = rcParams["draw_ci_50"]

    if ax is None:
        figure, ax = plt.subplots(figsize=(6, 3))

    # still need to fix the last dimension being one
    # if x.shape[0] != y.shape[-1]:
    #     log.exception(f"X rows and y rows do not match: {x.shape[0]} vs {y.shape[0]}")
    #     raise KeyError("Shape mismatch")

    if y.ndim == 2:
        data = np.median(y, axis=0)
    elif y.ndim == 1:
        data = y
    else:
        log.exception(f"y needs to be 1 or 2 dimensional, but has shape {y.shape}")
        raise KeyError("Shape mismatch")

    # ------------------------------------------------------------------------------ #
    # kwargs
    # ------------------------------------------------------------------------------ #

    if what is "data":
        if "color" not in kwargs:
            kwargs = dict(kwargs, color=rcParams["color_data"])
        if "marker" not in kwargs:
            kwargs = dict(kwargs, marker="d")
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs = dict(kwargs, ls="None")
    elif what is "fcast":
        if "color" not in kwargs:
            kwargs = dict(kwargs, color=rcParams["color_model"])
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs = dict(kwargs, ls=rcParams["fcast_ls"])
    elif what is "model":
        if "color" not in kwargs:
            kwargs = dict(kwargs, color=rcParams["color_model"])
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs = dict(kwargs, ls="-")

    # ------------------------------------------------------------------------------ #
    # plot
    # ------------------------------------------------------------------------------ #
    ax.plot(x, data, **kwargs)

    # overwrite some styles that do not play well with fill_between
    if "linewidth" in kwargs:
        del kwargs["linewidth"]
    if "marker" in kwargs:
        del kwargs["marker"]
    if "alpha" in kwargs:
        del kwargs["alpha"]
    if "label" in kwargs:
        del kwargs["label"]
    kwargs["lw"] = 0
    kwargs["alpha"] = 0.1

    if draw_ci_95 and y.ndim == 2:
        ax.fill_between(
            x,
            np.percentile(y, q=2.5, axis=0),
            np.percentile(y, q=97.5, axis=0),
            **kwargs,
        )

    if draw_ci_75 and y.ndim == 2:
        ax.fill_between(
            x,
            np.percentile(y, q=12.5, axis=0),
            np.percentile(y, q=87.5, axis=0),
            **kwargs,
        )

    del kwargs["alpha"]
    kwargs["alpha"] = 0.2

    if draw_ci_50 and y.ndim == 2:
        ax.fill_between(
            x,
            np.percentile(y, q=25.0, axis=0),
            np.percentile(y, q=75.0, axis=0),
            **kwargs,
        )

    # ------------------------------------------------------------------------------ #
    # formatting
    # ------------------------------------------------------------------------------ #
    _format_date_xticks(ax)

    return ax


def _get_array_from_trace_via_date(
    model, trace, var, start=None, end=None, dates=None,
):
    """
        Parameters
        ----------
        model : model instance

        trace : trace instance

        var : str
            the variable name in the trace

        start : datetime.datetime
            get all data for a range from `start` to `end`. (both boundary
            dates included)

        end :  datetime.datetime

        dates : list of datetime.datetime objects, optional
            the dates for which to get the data. Default: None, will return
            all available data.

        Returns
        -------
        data : nd array, 3 dim
            the elements from the trace matching the dates.
            dimensions are as follows
                0 samples, if no samples only one entry
                1 data with time matching the returned `dates` (if compatible variable)
                2 region, if no regions only one entry

        dates : pandas DatetimeIndex
            the matching dates. this is essnetially an array of dates than can be passed
            to matplotlib

        Example
        -------
        ```
            import covid19_inference as cov
            model, trace = cov.create_example_instance()
            y, x = cov.plot._get_array_from_trace_via_date(
                model, trace, "lambda_t", model.data_begin, model.data_end
            )
            ax = cov.plot._timeseries(x, y[:,:,0], what="model")
        ```
    """

    ref = model.sim_begin
    # the variable `new_cases` and some others (?) used to have different bounds
    # 20-05-27: not anymore, we made things consistent. let's remove this at some point
    # 20-05-29: still broken idk
    if "new_cases" in var:
        ref = model.data_begin

    if dates is None:
        if start is None:
            start = ref
        if end is None:
            end = model.sim_end
        dates = pd.date_range(start=start, end=end)
    else:
        assert start is None and end is None, "do not pass start/end with dates"
        # make sure its the right format
        dates = pd.DatetimeIndex(dates)

    indices = (dates - ref).days

    assert var in trace.varnames, "var should be in trace.varnames"
    assert np.all(indices >= 0), (
        "all dates should be after the model.sim_begin "
        + "(note that `new_cases` start at model.data_begin)"
    )
    assert np.all(indices < model.sim_len), "all dates should be before model.sim_end"

    # here we make sure that the returned array always has the same dimensions:
    if trace[var].ndim == 3:
        ret = trace[var][:, indices, :]
    elif trace[var].ndim == 2:
        ret = trace[var][:, indices]
        # ret = trace[var][:, indices, None]
        # 2020-05-06: jd and ps decided not to pad dimensions, not sure if it is more
        # confusing to have changing dimensions or dimensions that are not needed
        # in case of the non-hierarchical model
        # to access the region if you are not sure if it exists use an ellipsis:
        # region = ...
        # trace[var][:, indices, region]
        # will work fine if trace[var] is only 2-dimensional

    return ret, dates


def _new_cases_to_cum_cases(x, y, what, offset=0):
    """
        so this conversion got ugly really quickly.
        need to check dimensionality of y

        Parameters
        ----------
        x : pandas DatetimeIndex array
            will be padded accordingly

        y : 1d or 2d numpy array
            new cases matching dates in x.
            if 1d, we assume raw data (no samples)
            if 2d, we assume results from trace with 0th dim samples and 1st new cases
            matching x

        what : str
            dirty workaround to differntiate between traces and raw data
            "data" or "trace"

        offset : int or array like
            added to cum sum (should be the known cumulative case number at the
            first date of provided in x)

        Returns
        -------
        x_cum : pandas DatetimeIndex array
            dates of the cumulative cases

        y_cum : nd array
            cumulative cases matching x_cum and the dimension of input y

        Example
        -------
        ```
            cum_dates, cum_cases = _new_cases_to_cum_cases(new_dates, new_cases)
        ```
    """

    # things from the trace have the 0-th dimension for samples. raw data does not
    if what == "trace":
        y_cum = np.cumsum(y, axis=1) + offset
    elif what == "data":
        y_cum = np.cumsum(y, axis=0) + offset
    else:
        raise ValueError

    # example with offset = 0:
    # y_data new_cases [ 281  451  170 1597]
    # y_data cum_cases [ 281  732  902 2499]

    # so the cumulative used to be one day longer when applying the new cases to the
    # next day, then add a date at the end of the x axis
    # add one element using the existing frequency
    # x_cum = x.union(pd.DatetimeIndex([x[-1] + 1 * x.freq]))
    x_cum = x

    return x_cum, y_cum


# ------------------------------------------------------------------------------ #
# Distribution plotting
# ------------------------------------------------------------------------------ #


def _distribution(model, trace, key, ax=None, color=None, draw_prior=True):

    # check if model was hierarchical
    # if model.is_hierarchical
    # or like this
    # is_hc = False
    # for var in trace.varnames:
    #     if re.fullmatch('lambda_[0-9]+_L[0-9]', var) is not None:
    #         is_hc = True
    #     break

    # shape L2: samples, region, except sigma_L2 then no region
    # shape L1: samples

    if color is None:
        color = rcParams.color_model

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # todo: check compatible key before spending more time here
    data = trace[key]

    # apply additional transformations, if required
    if "transient_day" in key:
        # panda date time frame cannot do np.median, which we need
        # data = pd.to_datetime(data, origin=model.sim_begin, unit="D")
        data = _days_to_mpl_dates(data, origin=model.sim_begin)
    elif "weekend_factor_rad" == key:
        data = data / np.pi / 2 * 7

    ax.set_xlabel(_label_for_varname(key))
    ax.xaxis.set_label_position("top")

    # sometimes the bins are spread over very different x-ranges
    bins = 50
    if "lambda" in key or "mu" == key:
        bins = np.arange(0, 0.5 + 0.5 / bins, 0.5 / bins)

    # posteriors
    ax.hist(
        data,
        bins=bins,
        density=True,
        color=color,
        label="Posterior",
        alpha=0.7,
        zorder=-5,
    )

    # xlim
    if "lambda" in key or "mu" == key:
        ax.set_xlim(0, 0.5)
        ax.axvline(np.median(trace["mu"]), ls=":", color="black")
    elif "I_begin" == key:
        ax.set_xlim(0)
    elif "transient_len" in key:
        ax.set_xlim(0, 7)
    elif "transient_day" in key:
        # we will use this again later to align the printed median
        transient_day_md_mpl = np.median(data)
        ax.set_xlim([int(transient_day_md_mpl) - 4, int(transient_day_md_mpl) + 4])
        _format_date_xticks(ax)

    if draw_prior:
        # sample using pymc3. this avoids the headache of analytic solutions for
        # combined variables when we do not have analytic priors
        prior = pm.sample_prior_predictive(samples=500, model=model, var_names=[key])[
            key
        ]
        # smooth density from discrete histogram
        prior = stats.kde.gaussian_kde(prior)

        # may need to convert axes values, and restore xlimits after adding prior
        xlim = ax.get_xlim()
        x_for_ax = np.linspace(*xlim, num=100)
        x_for_pr = x_for_ax

        if "transient_day" in key:
            # cast datetime.datetime from model to mpl date format
            x_for_pr = x_for_ax - mpl.dates.date2num(model.sim_begin)
        if "weekend_factor_rad" == key:
            x_for_ax *= np.pi * 2 / 7

        ax.plot(
            x_for_ax,
            prior(x_for_pr),
            label="Prior",
            color=rcParams.color_prior,
            linewidth=3,
        )
        ax.set_xlim(*xlim)

    # add the overlay with median and CI values. these are two strings
    text_md = ""
    text_ci = ""
    if "lambda" in key or "mu" == key or "sigma_random_walk" == key:
        text_md, text_ci = _string_median_CI(data, prec=2)
    elif "transient_day" in key:
        # convert median from mpl date into datetime to adjust by month
        temp = mpl.dates.num2date(transient_day_md_mpl)
        data_shifted = data - mpl.dates.date2num(
            datetime.datetime(year=temp.year, month=temp.month, day=1)
        )
        # align 0 index with the first day of the month
        data_shifted = data_shifted + 1
        text_md, text_ci = _string_median_CI(data_shifted, prec=1,)
    else:
        text_md, text_ci = _string_median_CI(data, prec=1)

    text_md = _math_for_varname(key) + r"$ \simeq " + text_md + "$"

    # create the inset text elements, and we want a bounding box around the compound
    try:
        tel_md = ax.text(
            0.6,
            0.9,
            text_md,
            fontsize=12,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="center",
            zorder=100,
        )
        x_min, x_max, y_min, y_max = _get_mpl_text_coordinates(tel_md, ax)
        tel_ci = ax.text(
            0.6,
            y_min * 0.9,  # let's have a ten perecent margin or so
            text_ci,
            fontsize=9,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="center",
            zorder=101,
        )
        _add_mpl_rect_around_text(
            [tel_md, tel_ci], ax, facecolor="white", alpha=0.5, zorder=99,
        )
    except Exception as e:
        log.debug(f"unable to create inset with {key} value: {e}")

    # finalize
    ax.tick_params(labelleft=False)
    ax.set_rasterization_zorder(rcParams.rasterization_zorder)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if not "transient_day" in key:
        ax.locator_params(nbins=4)


def _label_for_varname(key):
    """
        get the label for trace variable names (e.g. placed on top of distributions)

        default for unknown keys is the key itself

        TODO
        ----
        add more parameters
    """
    res = key

    if re.fullmatch("lambda_0.*", key):
        res = "Initial rate"
    elif re.fullmatch("lambda.*", key):
        res = "Spreading rate " + _rx_cp_id(key)
    elif re.fullmatch("transient_day.*", key):
        res = "Change time " + _rx_cp_id(key)
    elif re.fullmatch("transient_len.*", key):
        res = "Change duration " + _rx_cp_id(key)
    elif re.fullmatch("delay.*", key):
        res = "Delay"
    elif re.fullmatch("mu.*", key):
        res = "Recovery rate"
    elif key == "sigma_obs":
        res = "Scale (width)\n of the likelihood"
    elif key == "I_begin":
        res = "Initial infections"
    return res


def _rx_cp_id(key):
    """
        get the change_point index from a compatible variable name
    """
    return re.search("_[0-9]+(_|$)", key).group().replace("_", "")


def _rx_hc_id(key):
    """
        get the L1 / L2 value of hierarchical variable name
    """
    if "_L1" in key:
        return 1
    elif "_L2" in key:
        return 2
    else:
        return None


def _math_for_varname(key):
    """
        get the math string for trace variable name, e.g. used to print the median
        representation.

        default for unknown keys is "$x$"

        TODO
        ----
        use regex
    """
    # default
    res = "x"

    # three options: unique, hierarchical and/or changepoint like
    is_un = True
    is_hc = False
    is_cp = False

    if "_L1" in key or "_L2" in key:
        is_hc = True
    if re.fullmatch(".+_[0-9]+.*", key):
        is_cp = True
    if is_cp or is_hc:
        is_un = False
    log.debug(f"_math_for_varname({key}): {int(is_un)} | {int(is_hc)} | {int(is_cp)}")

    # not unique
    if re.fullmatch("lambda.*", key):
        res = r"\lambda"
    elif re.fullmatch("transient_day.*", key):
        res = r"t"
    elif re.fullmatch("transient_len.*", key):
        res = r"\Delta t"
    elif re.fullmatch("sigma.*", key):
        # there is a lot of these guys. not making a distinction yet
        res = r"\sigma"
    elif re.fullmatch("delay.*", key):
        res = r"D"

    # unique keys
    if is_un:
        if "lambda_t" == key:
            res = r"\lambda_t"
        elif "I_begin" == key:
            res = r"I_0"
        elif "mu" == key:
            res = r"\mu"
        elif "sigma_obs" == key:
            res = r"\sigma"

    # change-point keys, give lower index
    if is_cp:
        # get cp index
        res = res + r"_{" + _rx_cp_id(key) + "}"

    # hierarchical, give upper index
    if is_hc:
        hc_suffix = ""
        if "_L1" in key:
            hc_suffix = r"^{1}"
        elif "_L2" in key:
            hc_suffix = r"^{2}"
        res = res + hc_suffix

    res = "$" + res + "$"

    return res


def _days_to_mpl_dates(days, origin):
    """
        convert days as number to matplotlib compatible date numbers.
        this is not the same as pandas dateindices, but numpy operations work on them

        Parameters
        ----------
        days : number, 1d array of numbers
            the day number to convert, e.g. integer values >= 0, one day per int

        origin : datetime.datetime
            the date object corresponding to day 0
    """
    try:
        return mpl.dates.date2num(
            [datetime.timedelta(days=float(date)) + origin for date in days]
        )
    except:
        return mpl.dates.date2num(datetime.timedelta(days=float(days)) + origin)


def _get_mpl_text_coordinates(text, ax):
    """
        helper to get coordinates of a text object in the coordinates of the
        axes element [0,1].
        used for the rectangle backdrop.

        Returns:
        x_min, x_max, y_min, y_max
    """
    fig = ax.get_figure()

    try:
        fig.canvas.renderer
    except Exception as e:
        log.debug(e)
        # otherwise no renderer, needed for text position calculation
        fig.canvas.draw()

    x_min = None
    x_max = None
    y_min = None
    y_max = None

    # get bounding box of text
    transform = ax.transAxes.inverted()
    try:
        bb = text.get_window_extent(renderer=fig.canvas.get_renderer())
    except:
        bb = text.get_window_extent()
    bb = bb.transformed(transform)
    x_min = bb.get_points()[0][0]
    x_max = bb.get_points()[1][0]
    y_min = bb.get_points()[0][1]
    y_max = bb.get_points()[1][1]

    return x_min, x_max, y_min, y_max


def _add_mpl_rect_around_text(text_list, ax, x_padding=0.05, y_padding=0.05, **kwargs):
    """
        add a rectangle to the axes (behind the text)

        provide a list of text elements and possible options passed to
        mpl.patches.Rectangle
        e.g.
        facecolor="grey",
        alpha=0.2,
        zorder=99,
    """

    x_gmin = 1
    y_gmin = 1
    x_gmax = 0
    y_gmax = 0

    for text in text_list:
        x_min, x_max, y_min, y_max = _get_mpl_text_coordinates(text, ax)
        if x_min < x_gmin:
            x_gmin = x_min
        if y_min < y_gmin:
            y_gmin = y_min
        if x_max > x_gmax:
            x_gmax = x_max
        if y_max > y_gmax:
            y_gmax = y_max

    # coords between 0 and 1 (relative to axes) add 10% margin
    y_gmin = np.clip(y_gmin - y_padding, 0, 1)
    y_gmax = np.clip(y_gmax + y_padding, 0, 1)
    x_gmin = np.clip(x_gmin - x_padding, 0, 1)
    x_gmax = np.clip(x_gmax + x_padding, 0, 1)
    rect = mpl.patches.FancyBboxPatch(
        (x_gmin, y_gmin),
        (x_gmax - x_gmin),
        (y_gmax - y_gmin),
        transform=ax.transAxes,
        boxstyle="round,pad=0,rounding_size=0.02",
        lw=0,
        **kwargs,
    )


    ax.add_patch(rect)


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
        date_format="%b %d",
        date_show_minor_ticks=True,
        rasterization_zorder=-1,
        draw_ci_95=True,
        draw_ci_75=False,
        draw_ci_50=False,
        color_model="tab:green",
        color_data="tab:blue",
        color_prior="#708090",
        color_annot="#646464",
        fcast_ls="--",
    )

    return par


def set_rcparams(par):
    """
        Set the rcparameters used for plotting. provided instance of `Param` has to have
        the following keys (attributes).

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

        draw_ci_50 : bool,
            For timeseries plots, indicate 50% Confidence interval via fill between.
            Default: False

        color_model : str,
            Base color used for model plots, mpl compatible color code "C0", "#303030"
            Default : "tab:green"

       color_data : str,
            Base color used for data
            Default : "tab:blue"

        color_annot : str,
            Color to use for annotations
            Default : "#646464"

        color_prior : str,
            Color to used for priors in distributions
            Default : "#708090"

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


def _format_k(prec):
    """
        format yaxis 10_000 as 10 k.
        _format_k(0)(1200, 1000.0) gives "1 k"
        _format_k(1)(1200, 1000.0) gives "1.2 k"
    """

    def inner(xval, tickpos):
        return f"${xval/1_000:.{prec}f}\,$k"

    return inner


def _format_date_xticks(ax, minor=None):
    # ensuring utf-8 helps on some setups
    locale.setlocale(locale.LC_ALL, rcParams.locale + ".UTF-8")
    ax.xaxis.set_major_locator(
        mpl.dates.WeekdayLocator(interval=1, byweekday=mpl.dates.SU)
    )
    if minor is None:
        # overwrite local argument with rc params only if default.
        minor = rcParams["date_show_minor_ticks"]
    if minor is True:
        ax.xaxis.set_minor_locator(mpl.dates.DayLocator())
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter(rcParams["date_format"]))


def _truncate_number(number, precision):
    return "{{:.{}f}}".format(precision).format(number)


def _string_median_CI(arr, prec=2):
    f_trunc = lambda n: _truncate_number(n, prec)
    med = f_trunc(np.median(arr))
    perc1, perc2 = (
        f_trunc(np.percentile(arr, q=2.5)),
        f_trunc(np.percentile(arr, q=97.5)),
    )
    # return "Median: {}\nCI: [{}, {}]".format(med, perc1, perc2)
    return f"{med}", f"[{perc1}, {perc2}]"


def _add_watermark(ax, mark="Model nach Dehning et al. arXiv:2004.01105"):
    """
        Add our arxive url to an axes as (upper right) title
    """

    # fig.text(
    #     pos[0],
    #     pos[1],
    #     "Dehning et al.",
    #     fontsize="medium",
    #     transform=  fig.transFigure,
    #     verticalalignment="top",
    #     horizontalalignment="right",
    #     color="#646464"
    #     # bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
    # )

    ax.set_title(mark, fontsize="small", loc="right", color="#646464")


# ------------------------------------------------------------------------------ #
# init
# ------------------------------------------------------------------------------ #
# set global parameter variables
rcParams = get_rcparams_default()
