# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-04-17 17:02:32
# @Last Modified: 2020-05-04 17:05:35
# ------------------------------------------------------------------------------ #
# I am reincluding the figure-plotting routines so the script is a bit more
# selfcontained. (also, figures.py is in a bad state currently)
# ------------------------------------------------------------------------------ #
# Modeling three different scenarios around easter. (one, two, or three changes)
# A) ORANGE: the german government plans to relax measures on 04/19.
# B) GREEN:  an improvement shortly after easter when the family visits are over
# ------------------------------------------------------------------------------ #


import sys
import copy
import datetime
import locale
import os

import numpy as np
import pymc3 as pm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

try:
    import covid19_inference as cov19
except ModuleNotFoundError:
    # imports are a bit picky when using native python
    # https://stackoverflow.com/regquestions/29548587/import-fails-when-running-python-as-script-but-not-in-ipython
    sys.path.append("")
    sys.path.append("../")
    sys.path.append("../../")
    sys.path.append("../../../")
    import covid19_inference as cov19

# ------------------------------------------------------------------------------ #
# global variables
# let's make them function arguments or attach them to model instances, soon
# ------------------------------------------------------------------------------ #


# save path for figures, ideally use absolute path. 'None' tries the repo figure folder
save_path = None

if save_path is None:
    try:
        # only works when called from python, not reliable in interactive ipython etc.
        os.chdir(os.path.dirname(__file__))
        save_path = "../../../figures/easter_"
    except:
        # assume base directory
        save_path = "../../../figures/easter_"

print(f"saving figures to {os.path.abspath(save_path)}")


confirmed_cases = cov19.get_jhu_confirmed_cases()

country = "Germany"
date_data_begin = datetime.datetime(2020, 3, 1)
date_data_end = cov19.get_last_date(confirmed_cases)
# date_data_end   = datetime.datetime(2020,4,17)

num_days_data = (date_data_end - date_data_begin).days

# diff_data_sim should be significantly larger than the expected delay, in
# order to always fit the same number of data points.
diff_data_sim = 16

# we only want to predict until March 01 for now
# num_days_future = (datetime.datetime(2020, 5, 2) - date_data_end).days
num_days_future = 42
date_begin_sim = date_data_begin - datetime.timedelta(days=diff_data_sim)
date_end_sim = date_data_end + datetime.timedelta(days=num_days_future)
num_days_sim = (date_end_sim - date_begin_sim).days

cases_obs = cov19.filter_one_country(
    confirmed_cases, country, date_data_begin, date_data_end
)

# styling for prior distributions
prio_style = {
    "color": "#708090",
    "linewidth": 3,
    "label": "Prior",
}
# styling for posterior distributions
post_style = {
    "density": True,
    "color": "tab:orange",
    "label": "Posterior",
    "zorder": -2,
}

# format the date on the x axis (see https://strftime.org/) example April 1 2020
# date_format = "%m/%d"  # 04/01
date_format = "%b %-d"  # Apr 1
# date_format = "%-d. %B" # 1. April
try:
    locale.setlocale(locale.LC_ALL, "en_US")
    # locale.setlocale(locale.LC_ALL,'de_DE')
except:
    pass

# whether to show the minor ticks (for every day)
date_show_minor_ticks = True


# set to None to keep everything a vector, with `-1` Posteriors are rastered (see above)
rasterization_zorder = -1

plot_par = dict()
plot_par["draw_insets_cases"] = False
plot_par["draw_ci_95"] = True
plot_par["draw_ci_75"] = False
plot_par["insets_only_two_ticks"] = True

# ------------------------------------------------------------------------------ #
# Formatting helpers
# ------------------------------------------------------------------------------ #

# format yaxis 10_000 as 10 k
format_k = lambda num, _: "${:.0f}\,$k".format(num / 1_000)

# format xaxis, ticks and labels
def format_date_xticks(ax, minor=None):
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
# Plotting helpers
# ------------------------------------------------------------------------------ #


def add_watermark(ax, mark="Dehning et al. arXiv:2004.01105"):
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


def create_figure_timeseries(
    trace,
    color="tab:green",
    save_to=None,
    num_days_futu_to_plot=18,
    y_lim_lambda=(-0.15, 0.45),
    plot_red_axis=True,
    axes=None,
    forecast_label="Forecast",
    add_more_later=False,
):
    """
        Used for the generation of the timeseries forecast figure around easter on the
        repo.

        Requires the plot_par dict to be set globally.

        Parameters
        ----------
        trace: trace instance
            needed for the data
        color: string
            main color to use, default "tab:green"
        save_to: string or None
            path where to save the figures. default: None, not saving figures
        num_days_futu_to_plot : int
            how many days to plot into the future (not exceeding simulation)
        y_lim_lambda : (float, float)
            min, max values for lambda effective. default (-0.15, 0.45)
        plot_red_axis : bool
            show the unconstrained constrained annotation in lambda panel
        axes : np.array of mpl axes
            provide an array of existing axes (from previously calling this function)
            to add more traces. Data will not be added again. Ideally call this first
            with `add_more_later=True`
        forecast_label : string
            legend label for the forecast, default: "Forecast"
        add_more_later : bool
            set this to true if you plan to add multiple models to the plot. changes the layout (and the color of the fit to past data)

        Returns
        -------
            fig : mpl figure
            axes : np array of mpl axeses (insets not included)

    """

    axes_provided = False
    if axes is not None:
        print("Provided axes, adding new content")
        axes_provided = True

    ylabel_new = f"Daily new reported\ncases in {country}"
    ylabel_cum = f"Total reported\ncases in {country}"
    ylabel_lam = f"Effective\ngrowth rate $\lambda^\\ast (t)$"

    pos_letter = (-0.25, 1)
    titlesize = 16
    insetsize = ("25%", "50%")
    # figsize = (4, 6)
    figsize = (6, 6)

    leg_loc = "upper left"
    if plot_par["draw_insets_cases"] == True:
        leg_loc = "upper right"

    new_c_ylim = [0, 9_000]
    new_c_insetylim = [50, 17_000]

    cum_c_ylim = [0, 300_000]
    cum_c_insetylim = [50, 300_000]

    color_futu = color
    color_past = color
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

    insets = []

    diff_to_0 = num_days_data + diff_data_sim

    # interval for the plots with forecast
    start_date = conv_time_to_mpl_dates(-len(cases_obs) + 2) + diff_to_0
    end_date = conv_time_to_mpl_dates(num_days_futu_to_plot) + diff_to_0
    mid_date = conv_time_to_mpl_dates(1) + diff_to_0

    # x-axis for dates, new_cases are one element shorter than cum_cases, use [1:]
    # 0 is the last recorded data point
    time_past = np.arange(-len(cases_obs) + 1, 1)
    time_futu = np.arange(0, num_days_futu_to_plot + 1)
    mpl_dates_past = conv_time_to_mpl_dates(time_past) + diff_to_0
    mpl_dates_futu = conv_time_to_mpl_dates(time_futu) + diff_to_0

    # --------------------------------------------------------------------------- #
    # prepare data
    # --------------------------------------------------------------------------- #
    # observed data, only one dim: [day]
    new_c_obsd = np.diff(cases_obs)
    cum_c_obsd = cases_obs

    # model traces, dims: [sample, day],
    new_c_past = trace["new_cases"][:, :num_days_data]
    new_c_futu = trace["new_cases"][
        :, num_days_data : num_days_data + num_days_futu_to_plot
    ]
    cum_c_past = np.cumsum(np.insert(new_c_past, 0, 0, axis=1), axis=1) + cases_obs[0]
    cum_c_futu = np.cumsum(np.insert(new_c_futu, 0, 0, axis=1), axis=1) + cases_obs[-1]

    # --------------------------------------------------------------------------- #
    # growth rate lambda*
    # --------------------------------------------------------------------------- #
    ax = axes[0]
    mu = trace["mu"][:, np.newaxis]
    lambda_t = trace["lambda_t"][
        :, diff_data_sim : diff_data_sim + num_days_data + num_days_futu_to_plot
    ]
    ax.plot(
        np.concatenate([mpl_dates_past[1:], mpl_dates_futu[1:]]),
        np.median(lambda_t - mu, axis=0),
        color=color_futu,
        linewidth=2,
    )
    if plot_par["draw_ci_95"] == True:
        ax.fill_between(
            np.concatenate([mpl_dates_past[1:], mpl_dates_futu[1:]]),
            np.percentile(lambda_t - mu, q=2.5, axis=0),
            np.percentile(lambda_t - mu, q=97.5, axis=0),
            alpha=0.15,
            color=color_futu,
            lw=0,
        )

    ax.set_ylabel(ylabel_lam)
    ax.set_ylim(*y_lim_lambda)
    # xlim at the end of the function, synchornized axes

    if not axes_provided:
        ax.text(
            pos_letter[0], pos_letter[1], "A", transform=ax.transAxes, size=titlesize
        )
        ax.hlines(0, start_date, end_date, linestyles=":")
        delay = matplotlib.dates.date2num(date_data_end) - np.percentile(
            trace.delay, q=75
        )
        if plot_red_axis:
            ax.vlines(delay, -10, 10, linestyles="-", colors=["#646464"])
            ax.text(
                delay + 1.5,
                0.4,
                "unconstrained due\nto reporting delay",
                color="#646464",
                verticalalignment="top",
            )
            ax.text(
                delay - 1.5,
                0.4,
                "constrained\nby data",
                color="#646464",
                horizontalalignment="right",
                verticalalignment="top",
            )

    # --------------------------------------------------------------------------- #
    # New cases, lin scale first
    # --------------------------------------------------------------------------- #
    ax = axes[1]
    if not axes_provided:
        ax.text(
            pos_letter[0], pos_letter[1], "B", transform=ax.transAxes, size=titlesize
        )
        ax.plot(
            mpl_dates_past[1:],
            new_c_obsd,
            "d",
            label="Data",
            markersize=4,
            color="tab:blue",
            zorder=5,
        )
        ax.plot(
            mpl_dates_past[1:],
            np.median(new_c_past, axis=0),
            "-",
            color=color_past,
            linewidth=1.5,
            label="Fit",
            zorder=10,
        )
        if plot_par["draw_ci_95"] == True:
            ax.fill_between(
                mpl_dates_past[1:],
                np.percentile(new_c_past, q=2.5, axis=0),
                np.percentile(new_c_past, q=97.5, axis=0),
                alpha=0.1,
                color=color_past,
                lw=0,
            )
        # dummy element to separate forecasts
        if add_more_later:
            ax.plot(
                [],
                [],
                "-",
                linewidth=0,
                label="Forecasts:",
                # fontweight="bold"
            )

    ax.plot(
        mpl_dates_futu[1:],
        np.median(new_c_futu, axis=0),
        "--",
        color=color_futu,
        linewidth=3,
        label=forecast_label,
    )
    if plot_par["draw_ci_95"] == True:
        ax.fill_between(
            mpl_dates_futu[1:],
            np.percentile(new_c_futu, q=2.5, axis=0),
            np.percentile(new_c_futu, q=97.5, axis=0),
            alpha=0.1,
            color=color_futu,
            lw=0,
        )
    if plot_par["draw_ci_75"] == True:
        ax.fill_between(
            mpl_dates_futu[1:],
            np.percentile(new_c_futu, q=12.5, axis=0),
            np.percentile(new_c_futu, q=87.5, axis=0),
            alpha=0.2,
            color=color_futu,
            lw=0,
        )
    ax.set_ylabel(ylabel_new)
    ax.set_ylim(new_c_ylim)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_k))

    # NEW CASES LOG SCALE, skip forecast
    if plot_par["draw_insets_cases"] == True:
        ax = inset_axes(
            ax, width=insetsize[0], height=insetsize[1], loc=2, borderpad=0.75
        )
        insets.append(ax)
        if not axes_provided:
            ax.plot(
                mpl_dates_past[1:],
                new_c_obsd,
                "d",
                markersize=2,
                label="Data",
                zorder=5,
            )
        ax.plot(
            mpl_dates_past[1:],
            np.median(new_c_past, axis=0),
            "-",
            color=color_past,
            label="Fit",
            zorder=10,
        )
        if plot_par["draw_ci_95"] == True:
            ax.fill_between(
                mpl_dates_past[1:],
                np.percentile(new_c_past, q=2.5, axis=0),
                np.percentile(new_c_past, q=97.5, axis=0),
                alpha=0.1,
                color=color_past,
                lw=0,
            )
        ax.set_yticks([1e1, 1e2, 1e3, 1e4, 1e5])
        ax.set_ylim(new_c_insetylim)

    # --------------------------------------------------------------------------- #
    # Total cases, lin scale first
    # --------------------------------------------------------------------------- #
    ax = axes[2]
    if not axes_provided:
        ax.text(
            pos_letter[0], pos_letter[1], "C", transform=ax.transAxes, size=titlesize
        )
        ax.plot(
            mpl_dates_past[:],
            cum_c_obsd,
            "d",
            label="Data",
            markersize=4,
            color="tab:blue",
            zorder=5,
        )
        ax.plot(
            mpl_dates_past[:],
            np.median(cum_c_past, axis=0),
            "-",
            color=color_past,
            linewidth=1.5,
            label="Fit",
            zorder=10,
        )
        if plot_par["draw_ci_95"] == True:
            ax.fill_between(
                mpl_dates_past[:],
                np.percentile(cum_c_past, q=2.5, axis=0),
                np.percentile(cum_c_past, q=97.5, axis=0),
                alpha=0.1,
                color=color_past,
                lw=0,
            )
        # dummy element to separate forecasts
        if add_more_later:
            ax.plot(
                [],
                [],
                "-",
                linewidth=0,
                label="Forecasts:",
                # fontweight="bold"
            )

    ax.plot(
        mpl_dates_futu[1:],
        np.median(cum_c_futu[:, 1:], axis=0),
        "--",
        color=color_futu,
        linewidth=3,
        label=f"{forecast_label}",
    )
    if plot_par["draw_ci_95"] == True:
        ax.fill_between(
            mpl_dates_futu[1:],
            np.percentile(cum_c_futu[:, 1:], q=2.5, axis=0),
            np.percentile(cum_c_futu[:, 1:], q=97.5, axis=0),
            alpha=0.1,
            color=color_futu,
            lw=0,
        )
    if plot_par["draw_ci_75"] == True:
        ax.fill_between(
            mpl_dates_futu[1:],
            np.percentile(cum_c_futu[:, 1:], q=12.5, axis=0),
            np.percentile(cum_c_futu[:, 1:], q=87.5, axis=0),
            alpha=0.2,
            color=color_futu,
            lw=0,
        )
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel_cum)
    ax.set_ylim(cum_c_ylim)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_k))

    # Total CASES LOG SCALE, skip forecast
    if plot_par["draw_insets_cases"] == True:
        ax = inset_axes(
            ax, width=insetsize[0], height=insetsize[1], loc=2, borderpad=0.75
        )
        insets.append(ax)
        if not axes_provided:
            ax.plot(
                mpl_dates_past[:], cum_c_obsd, "d", markersize=2, label="Data", zorder=5
            )
        ax.plot(
            mpl_dates_past[:],
            np.median(cum_c_past, axis=0),
            "-",
            color=color_past,
            label="Fit",
            zorder=10,
        )
        if plot_par["draw_ci_95"] == True:
            ax.fill_between(
                mpl_dates_past[:],
                np.percentile(cum_c_past, q=2.5, axis=0),
                np.percentile(cum_c_past, q=97.5, axis=0),
                alpha=0.1,
                color=color_past,
                lw=0,
            )
        ax.set_yticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
        ax.set_ylim(cum_c_insetylim)

    # --------------------------------------------------------------------------- #
    # Finalize
    # --------------------------------------------------------------------------- #

    for ax in axes:
        ax.set_rasterization_zorder(rasterization_zorder)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlim(start_date, end_date)
        format_date_xticks(ax)

        # biweekly, remove every second element
        if not axes_provided:
            for label in ax.xaxis.get_ticklabels()[1::2]:
                label.set_visible(False)

    for ax in insets:
        ax.set_xlim(start_date, mid_date)
        ax.yaxis.tick_right()
        ax.set_yscale("log")
        if plot_par["insets_only_two_ticks"] is True:
            format_date_xticks(ax, minor=False)
            for label in ax.xaxis.get_ticklabels()[1:-1]:
                label.set_visible(False)
            print(ax.xticks)
        else:
            format_date_xticks(ax)
            for label in ax.xaxis.get_ticklabels()[1:-1]:
                label.set_visible(False)

    # legend
    ax = axes[2]
    ax.legend(loc=leg_loc)
    ax.get_legend().get_frame().set_linewidth(0.0)
    ax.get_legend().get_frame().set_facecolor("#F0F0F0")

    add_watermark(axes[1])

    fig.suptitle(
        # using script run time. could use last data point though.
        f"Latest forecast\n({datetime.datetime.now().strftime('%Y/%m/%d')})",
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
            save_to + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0,
        )
        plt.savefig(
            save_to + ".png", dpi=300, bbox_inches="tight", pad_inches=0,
        )

    # add insets to returned axes. maybe not, general axes style would be applied
    # axes = np.append(axes, insets)

    return fig, axes


def get_label_dict():
    labels = dict()
    labels["mu"] = f"Recovery rate"
    labels["delay"] = f"Reporting delay"
    labels["I_begin"] = f"Initial infections"
    labels["sigma_obs"] = f"Scale (width)\nof the likelihood"
    labels["lambda_0"] = f"Initial rate"
    labels["lambda_1"] = f"Spreading rates"
    labels["transient_begin_0"] = f"Change times"
    labels["transient_len_0"] = f"Change duration"
    labels["E_begin_scale"] = "Initial scale\nof exposed"
    labels["median_incubation"] = "Median\nincubation delay"
    labels["sigma_random_walk"] = "Std. of\nrandom walk"
    labels["weekend_factor"] = "Factor\nweekends discounted"
    labels["offset_modulation_rad"] = "Offset from sunday\nof the modulation"
    return labels


def labels(key):
    if key == "mu":
        return f"Recovery rate"


def get_mpl_text_coordinates(text, ax):
    """
        helper to get a coordinates of a text object in the coordinates of the
        axes element.
        used for the rectangle backdrop.

        Returns:
        x_min, x_max, y_min, y_max
    """
    fig = ax.get_figure()

    try:
        fig.canvas.renderer
    except Exception as e:
        # print(e)
        # otherwise no renderer, needed for text position calculation
        fig.canvas.draw()

    x_min = None
    x_max = None
    y_min = None
    y_max = None

    # get bounding box of text
    transform = ax.transAxes.inverted()
    bb = text.get_window_extent(renderer=fig.canvas.get_renderer())
    bb = bb.transformed(transform)
    x_min = bb.get_points()[0][0]
    x_max = bb.get_points()[1][0]
    y_min = bb.get_points()[0][1]
    y_max = bb.get_points()[1][1]

    return x_min, x_max, y_min, y_max


def add_mpl_rect_around_text(text_list, ax, **kwargs):
    """
        add a rectangle to the axes (behind the text)

        provide a list of text elements and possible options passed to patches.Rectangle

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
        x_min, x_max, y_min, y_max = get_mpl_text_coordinates(text, ax)
        if x_min < x_gmin:
            x_gmin = x_min
        if y_min < y_gmin:
            y_gmin = y_min
        if x_max > x_gmax:
            x_gmax = x_max
        if y_max > y_gmax:
            y_gmax = y_max

    pad_percent = 10
    # x_gmin *= 1 - pad_percent / 100
    y_gmin *= 1 - pad_percent / 100
    # x_gmax *= 1 + pad_percent / 100
    y_gmax *= 1 + pad_percent / 100

    rect = patches.Rectangle(
        (x_gmin, y_gmin),
        x_gmax - x_gmin,
        y_gmax - y_gmin,
        transform=ax.transAxes,
        **kwargs,
    )

    ax.add_patch(rect)


def create_figure_distributions(
    model,
    trace,
    save_to=None,
    additional_insets=None,
    xlim_lambda=(0, 0.53),
    color="tab:green",
    num_changepoints=3,
    xlim_tbegin=4,
):
    """
        create the distribution overview plot.
    """
    if additional_insets is None:
        additional_insets = {}
    colors = ["#708090", color]

    additional_dists = len(additional_insets)
    num_columns = int(np.ceil((additional_dists + 14) / 5))
    num_rows = num_changepoints + 2
    width_col = 4.5 / 3 * num_columns
    height_fig = 6 if not num_changepoints == 1 else 5
    fig, axes = plt.subplots(
        num_rows, num_columns, figsize=(width_col, 6), constrained_layout=True
    )

    xlim_transt = (0, 7)
    xlim_tbegin = xlim_tbegin  # median plus minus x days

    axpos = dict()
    letters = dict()

    # leave away the closing doller, we add it later
    insets = dict()
    insets["lambda_0"] = r"$\lambda_0 \simeq "
    for i in range(0, num_changepoints):
        insets[f"lambda_{i+1}"] = f"$\lambda_{i+1} \simeq "
        insets[f"transient_begin_{i}"] = f"$t_{i} \simeq "
        insets[f"transient_len_{i}"] = f"$\Delta t_{i} \simeq "

    insets["mu"] = r"$\mu \simeq "
    insets["delay"] = r"$D \simeq "
    insets["sigma_obs"] = r"$\sigma \simeq "
    insets["I_begin"] = r"$I_0 \simeq "

    for key, inset in additional_insets.items():
        insets[key] = inset

    # layout 2
    pos_letter = (-0.2, 1.1)
    labels = get_label_dict()
    axpos["lambda_0"] = axes[1][0]
    for i_cp in range(0, num_changepoints):
        axpos["lambda_{}".format(i_cp + 1)] = axes[i_cp + 2][0]
        axpos["transient_begin_{}".format(i_cp)] = axes[i_cp + 2][1]
        axpos["transient_len_{}".format(i_cp)] = axes[i_cp + 2][2]

    axpos["mu"] = axes[0][0]
    axpos["delay"] = axes[1][2]
    axpos["I_begin"] = axes[1][1]
    axpos["sigma_obs"] = axes[0][1]

    letters["lambda_0"] = r"E"
    letters["lambda_1"] = r"F"
    letters["mu"] = r"D"

    i = num_rows - 1
    for i, (key, inset) in enumerate(additional_insets.items()):
        print(f"additional insets: {key}")
        col = int(np.floor((i + 14) / num_rows))
        row = i % num_rows
        axpos[key] = axes[row][col]
    for i in range(i + 1, num_rows):
        axes[i][col].set_visible(False)
    if not len(additional_insets) % num_rows == 1:
        axpos["legend"] = axes[0][num_columns - 1]

    # render panels
    for key in axpos.keys():
        if "legend" in key:
            continue

        data = trace[key]
        if "transient_begin" in key:
            data = conv_time_to_mpl_dates(trace[key])
        elif "weekend_factor_rad" == key:
            data = data / np.pi / 2 * 7

        ax = axpos[key]
        # using .get returns None when the key is not in the dict, and avoids KeyError
        ax.set_xlabel(labels.get(key))
        ax.xaxis.set_label_position("top")

        # make some bold
        if key == "lambda_1" or key == "transient_begin_0" or key == "transient_len_0":
            ax.set_xlabel(labels.get(key), fontweight="bold")

        # posteriors
        ax.hist(
            data,
            bins=50,
            density=True,
            color=colors[1],
            label="Posterior",
            alpha=0.7,
            zorder=-5,
        )

        # xlim
        if "lambda" in key or "mu" == key:
            ax.set_xlim(xlim_lambda)
            ax.axvline(np.median(trace["mu"]), ls=":", color="black")
        elif "I_begin" == key:
            ax.set_xlim(0)
        elif "transient_len" in key:
            ax.set_xlim(xlim_transt)
        elif "transient_begin" in key:
            md = np.median(data)
            ax.set_xlim([int(md) - xlim_tbegin, int(md) + xlim_tbegin - 1])
            format_date_xticks(ax)

        # priors
        limits = ax.get_xlim()
        x_for_ax = np.linspace(*limits, num=100)
        x_for_pr = x_for_ax
        if "transient_begin" in key:
            beg_x = matplotlib.dates.num2date(x_for_ax[0])
            diff_dates_x = (beg_x.replace(tzinfo=None) - date_begin_sim).days
            x_for_pr = x_for_ax - x_for_ax[0] + diff_dates_x
        if "weekend_factor_rad" == key:
            x_for_ax *= np.pi * 2 / 7
        ax.plot(
            x_for_ax,
            cov19.plotting.get_prior_distribution(model, x_for_pr, key),
            label="Prior",
            color=colors[0],
            linewidth=3,
        )
        ax.set_xlim(*limits)

        # letters
        if key in letters.keys():
            ax.text(
                pos_letter[0],
                pos_letter[1],
                letters[key],
                transform=ax.transAxes,
                size=14,
                horizontalalignment="left",
            )

        # median
        global text
        if "lambda" in key or "mu" == key or "sigma_random_walk" == key:
            text = print_median_CI(data, prec=2)
        elif "transient_begin" in key:
            text = print_median_CI(
                data - matplotlib.dates.date2num(date_data_begin) + 1, prec=1
            )
        else:
            text = print_median_CI(data, prec=1)

        if key in insets.keys():
            # strip everything except the median value
            text = text.replace("Median: ", "").replace("CI: ", "")
            md = text.split("\n")[0]
            ci = text.split("\n")[1]

            # create the inset text and we want a bounding box around the compound
            # text = insets[key] + md + "$" + "\n" + r"$\,$"
            text = insets[key] + md + "$"
            t_xl = ax.text(
                0.6,
                0.9,
                text,
                fontsize=12,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="center",
                # bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
                zorder=100,
            )

            x_min, x_max, y_min, y_max = get_mpl_text_coordinates(t_xl, ax)

            t_sm = ax.text(
                0.6,
                y_min,
                ci,
                fontsize=9,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="center",
                # bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
                zorder=101,
            )

            add_mpl_rect_around_text(
                [t_xl, t_sm], ax, facecolor="white", alpha=0.5, zorder=99,
            )

    # legend
    if "legend" in axpos:
        ax = axpos["legend"]
        ax.set_axis_off()
        ax.plot([], [], color=colors[0], linewidth=3, label="Prior")
        ax.hist([], color=colors[1], label="Posterior")
        ax.legend(loc="center left")
        ax.get_legend().get_frame().set_linewidth(0.0)
        ax.get_legend().get_frame().set_facecolor("#F0F0F0")

    # dirty hack to get some space at the bottom to align with timeseries
    if not num_changepoints == 1:
        axes[-1][0].xaxis.set_label_position("bottom")
        axes[-1][0].set_xlabel(r"$\,$")

    for jdx, ax_row in enumerate(axes):
        for idx, ax in enumerate(ax_row):
            if idx == 0 and jdx == 0:
                ax.set_ylabel("Density")
            ax.tick_params(labelleft=False)
            ax.locator_params(nbins=4)
            ax.set_rasterization_zorder(rasterization_zorder)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

    # plt.subplots_adjust(wspace=0.2, hspace=0.9)

    if save_to is not None:
        plt.savefig(save_to + ".pdf", bbox_inches="tight", pad_inches=0, dpi=300)
        plt.savefig(save_to + ".png", bbox_inches="tight", pad_inches=0, dpi=300)


# ------------------------------------------------------------------------------ #
# define change points for the three models
# ------------------------------------------------------------------------------ #

change_points_B = [
    dict(  # initial change from lambda 0 to lambda 1
        pr_mean_date_begin_transient=datetime.datetime(2020, 3, 9), pr_median_lambda=0.2
    ),
    dict(  # towards lambda 2
        pr_mean_date_begin_transient=datetime.datetime(2020, 3, 16),
        pr_sigma_date_begin_transient=1,
        pr_median_lambda=1 / 8,
        pr_sigma_lambda=0.2,
    ),
    dict(  # towards lambda 3, finally below-zero growth
        pr_mean_date_begin_transient=datetime.datetime(2020, 3, 23),
        pr_sigma_date_begin_transient=1,
        pr_median_lambda=1 / 8 / 2,
        pr_sigma_lambda=0.2,
    ),
    dict(  # back to lambda 2 around easter.
        pr_mean_date_begin_transient=datetime.datetime(2020, 4, 7),
        pr_sigma_date_begin_transient=1,
        pr_median_lambda=0.13,  # using the inferred value as of 04/20 and wider prior
        pr_sigma_lambda=0.3,
    ),
    dict(  # shortly after easter, people stop visiting and we might go back to lambda 3
        pr_mean_date_begin_transient=datetime.datetime(2020, 4, 12),
        pr_sigma_date_begin_transient=1,
        pr_median_lambda=0.08,  # using the inferred value as of 04/20 and wider prior
        pr_sigma_lambda=0.3,
    ),
]


# add another change point
change_points_A = copy.deepcopy(change_points_B)
change_points_A.append(
    dict(  # German government will relax restrictions on April 19
        pr_mean_date_begin_transient=datetime.datetime(2020, 4, 19),
        pr_sigma_date_begin_transient=1,
        pr_median_lambda=0.13,  # using the inferred value as of 04/20 and wider prior
        pr_sigma_lambda=0.3,
    ),
)

# ------------------------------------------------------------------------------ #
# set and run the three models
# ------------------------------------------------------------------------------ #

model_A = cov19.SIR_with_change_points(
    np.diff(cases_obs),
    change_points_A,
    date_begin_sim,
    num_days_sim,
    diff_data_sim,
    N=83e6,
    weekends_modulated=True,
    weekend_modulation_type="abs_sine",
)
model_B = cov19.SIR_with_change_points(
    np.diff(cases_obs),
    change_points_B,
    date_begin_sim,
    num_days_sim,
    diff_data_sim,
    N=83e6,
    weekends_modulated=True,
    weekend_modulation_type="abs_sine",
)

trace_A = pm.sample(model=model_A, init="advi", tune=1000, draws=1000)
print("Finished simulations for model A")
trace_B = pm.sample(model=model_B, init="advi", tune=1000, draws=1000)
print("Finished simulations for model B")


# ------------------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------------------ #

create_figure_timeseries(
    trace_A,
    color="tab:orange",
    num_days_futu_to_plot=num_days_future,
    save_to=save_path + "ts_A",
)
create_figure_timeseries(
    trace_B,
    color="tab:green",
    num_days_futu_to_plot=num_days_future,
    save_to=save_path + "ts_B",
)

# have a combined plot, by providing the axes elements
fig, axes = create_figure_timeseries(
    trace_A,
    color="tab:orange",
    num_days_futu_to_plot=num_days_future,
    forecast_label="Scenario A",
    add_more_later=True,
)
create_figure_timeseries(
    trace_B,
    color="tab:green",
    num_days_futu_to_plot=num_days_future,
    axes=axes,
    save_to=save_path + "ts_merged",
    forecast_label="Scenario B",
)


create_figure_distributions(
    model_A,
    trace_A,
    color="tab:orange",
    num_changepoints=len(change_points_A),
    save_to=save_path + "dist_A",
)
create_figure_distributions(
    model_B,
    trace_B,
    color="tab:green",
    num_changepoints=len(change_points_B),
    save_to=save_path + "dist_B",
)
