import logging
import datetime
import locale
import copy
import re
import sys
import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats

log = logging.getLogger(__name__)
try:
    import covid19_inference_new as cov19
except ModuleNotFoundError:
    sys.path.append("../..")
    import covid19_inference_new as cov19


def create_plot_scenarios(
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
    Creates a timeseries overview:
        - lambda
        - new_cases (rolling average 7 days and normalized by 1.000.000 pop)
    """
    # ------------------------------------------------------------------------------ #
    # Preliminary plotting configs
    # ------------------------------------------------------------------------------ #

    # Get rcParams
    rcParams = cov19.plot.rcParams

    # Some spezial changes
    figsize = (6, 6)
    ylim_lam = [-0.15, 0.15]

    # Locales i.e. which language have we defined!
    label_y_new = f"Reported cases\nper million."
    label_y_cum = f"Total\nreported cases"
    label_y_lam = f"Effective\ngrowth rate $\lambda^\\ast (t)$"
    label_y_R = f"Reproduction number\n$R(t)$"
    label_leg_data = "Data"
    label_leg_dlim = f"Updated on\n{datetime.datetime.now().strftime('%Y/%m/%d')}"
    label_forecast_line = ""
    label_x = "Date"

    if rcParams["locale"].lower() == "de_de":
        label_y_new = f"Fallzahlen\npro 1.000.000 EW"
        label_y_cum = f"Gesamtzahl\ngemeldeter Fälle"
        label_y_lam = f"Effektive\nWachstumsrate"
        label_y_R = f"Reproduktionszahl"
        label_leg_data = "Daten"
        label_leg_dlim = (
            f"Aktualisiert am\n{datetime.datetime.now().strftime('%-d. %B %Y')}"
        )
        label_x = "Datum"

    letter_kwargs = dict(x=-0.25, y=1, size="x-large")

    # per default we assume no hierarchical
    if region is None:
        region = ...

    # Check if an axes is provided
    axes_provided = False
    if axes is not None:
        log.debug("Provided axes, adding new content")
        axes_provided = True

    # Set local vars for colors
    color_data = rcParams.color_data
    color_past = rcParams.color_model
    color_fcast = rcParams.color_model
    color_annot = rcParams.color_annot

    # Check if we supplied an additional color
    if color is not None:
        color_past = color
        color_fcast = color

    # Get figure if we provided axes
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

    # Get model start and end date
    if start is None:
        start = model.data_begin
    if end is None:
        end = model.sim_end

    # insets are not reimplemented yet
    insets = []
    insets_only_two_ticks = True
    draw_insets = False

    # ------------------------------------------------------------------------------ #
    # lambda*, effective growth rate (Twinx plot with R)
    # ------------------------------------------------------------------------------ #

    # Get data for mu and lambda from trace
    mu = trace["mu"][:, None]
    lambda_t, x = cov19.plot._get_array_from_trace_via_date(model, trace, "lambda_t")
    y = lambda_t[:, :, region] - mu

    # Plot data
    cov19.plot._timeseries(x=x, y=y, ax=axes[0], what="model", color=color_fcast)
    axes[0].set_ylabel(label_y_lam)
    axes[0].set_ylim(ylim_lam)

    # Plot letter only once
    if not axes_provided:
        axes[0].text(s="A", transform=axes[0].transAxes, **letter_kwargs)
        axes[0].hlines(0, x[0], x[-1], linestyles=":")

    # --------------------------------------------------------------------------- #
    # New cases, lin scale first
    # --------------------------------------------------------------------------- #
    ax = axes[1]

    # Get new infered new cases from trace
    y_past, x_past = cov19.plot._get_array_from_trace_via_date(
        model, trace, "new_cases", model.data_begin, model.data_end
    )
    # Calculate rolling average
    y_past = pd.DataFrame(y_past).rolling(7, axis=1).mean().to_numpy() / 83.02e6 * 1e6
    # Select region if hierachical
    y_past = y_past[:, :, region]

    # Get new_cases_data i.e. observed cases and datetime range
    y_data = model.new_cases_obs[:, region] / 83.02e6 * 1e6
    x_data = pd.date_range(start=model.data_begin, end=model.data_end)

    # Get model prediction from trace
    y_fcast, x_fcast = cov19.plot._get_array_from_trace_via_date(
        model,
        trace,
        "new_cases",
        model.fcast_begin - datetime.timedelta(days=7),
        model.fcast_end,
    )
    y_fcast = pd.DataFrame(y_fcast).rolling(7, axis=1).mean().to_numpy() / 83.02e6 * 1e6
    y_fcast = y_fcast[:, :, region]

    # Plot

    # infered
    cov19.plot._timeseries(
        x=x_past, y=y_past, ax=axes[1], what="model", color=color_past,  # label="Fit",
    )

    # prediction
    cov19.plot._timeseries(
        x=x_fcast,
        y=y_fcast,
        ax=axes[1],
        what="fcast",
        color=color_fcast,
        label=f"{forecast_label}",
    )

    # Data and letter only once
    # !!For now I disabled the data plotting!!
    if not axes_provided:
        ax.text(s="B", transform=ax.transAxes, **letter_kwargs)

        """cov19.plot._timeseries(
            x=x_data,
            y=y_data,
            ax=ax,
            what="data",
            color=color_data,
            zorder=5,
            label=label_leg_data,
        )"""

    axes[1].set_ylabel(label_y_new)
    axes[1].set_xlabel(label_x)
    prec = 1.0 / (np.log10(ax.get_ylim()[1]) - 2.5)
    if prec < 2.0 and prec >= 0:
        axes[1].yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(_format_k(int(prec)))
        )

    # ------------------------------------------------------------------------------ #
    # total cases, still needs work because its not in the trace, we cant plot it
    # due to the lacking offset from new to cumulative cases, we cannot calculate
    # either.
    # ------------------------------------------------------------------------------ #
    ax = axes[2]

    y_past, x_past = cov19.plot._get_array_from_trace_via_date(
        model, trace, "new_cases", model.data_begin, model.data_end
    )
    y_past = y_past[:, :, region]

    y_data = model.new_cases_obs[:, region] / 83.02e6 * 1e6
    x_data = pd.date_range(start=model.data_begin, end=model.data_end)

    x_data, y_data = cov19.plot._new_cases_to_cum_cases(x_data, y_data, "data", offset)
    x_past, y_past = cov19.plot._new_cases_to_cum_cases(x_past, y_past, "trace", offset)

    # data points and annotations, draw only once
    if not axes_provided:
        ax.text(s="C", transform=ax.transAxes, **letter_kwargs)
        cov19.plot._timeseries(
            x=x_data,
            y=y_data,
            ax=ax,
            what="data",
            color=color_data,
            zorder=5,
            label=label_leg_data,
        )
        # model fit
        cov19.plot._timeseries(
            x=x_past, y=y_past, ax=ax, what="model", color=color_past,  # label="Fit",
        )
        if add_more_later:
            # dummy element to separate forecasts
            ax.plot(
                [], [], "-", linewidth=0, label=forecast_heading,
            )

    # model fcast, needs to start one day later, too. use the end date we got before
    y_fcast, x_fcast = cov19.plot._get_array_from_trace_via_date(
        model, trace, "new_cases", model.fcast_begin, model.fcast_end
    )
    y_fcast = y_fcast[:, :, region]

    # offset according to last cumulative model point
    x_fcast, y_fcast = cov19.plot._new_cases_to_cum_cases(
        x_fcast, y_fcast, "trace", y_past[:, -1, None]
    )

    cov19.plot._timeseries(
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
        cov19.plot._format_date_xticks(ax)

        # biweekly, remove every second element

    for ax in insets:
        ax.set_xlim(start, model.data_end)
        ax.yaxis.tick_right()
        ax.set_yscale("log")
        if insets_only_two_ticks is True:
            cov19.plot.format_date_xticks(ax, minor=False)
            for label in ax.xaxis.get_ticklabels()[1:-1]:
                label.set_visible(False)
            print(ax.xticks)
        else:
            cov19.plot.format_date_xticks(ax)
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
        cov19.plot._add_watermark(
            axes[1], mark="Model nach Dehning et al. 10.1126/science.abb9789"
        )

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

    # add insets to returned axes. maybe not, general axes style would be applied
    # axes = np.append(axes, insets)

    return fig, axes


def _format_k(prec):
    """"""

    def inner(xval, tickpos):
        return f"${xval:.{prec}f}\,$"

    return inner
