# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-04-17 17:02:32
# @Last Modified: 2020-04-17 18:03:02
# ------------------------------------------------------------------------------ #
# I am reincluding the figure-plotting routines so the script is a bit more
# selfcontained. (also, figures.py is in a bad state currently)
# ------------------------------------------------------------------------------ #
# Modeling three different scenarios around easter. (one, two, or three changes)
# A) just a change around easter (an increaste in spreading)
# B) possibly an improvement shortly after easter when the visits are ofter
# C) the german government plans to relax measures on 04/19.
# ------------------------------------------------------------------------------ #


import sys
import copy
import datetime

import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

try:
    import covid19_inference as cov19
except ModuleNotFoundError:
    sys.path.append("../")
    sys.path.append("../../")
    sys.path.append("../../../")
    import covid19_inference as cov19

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
num_days_future = (datetime.datetime(2020, 5, 2) - date_data_end).days
date_begin_sim = date_data_begin - datetime.timedelta(days=diff_data_sim)
date_end_sim = date_data_end + datetime.timedelta(days=num_days_future)
num_days_sim = (date_end_sim - date_begin_sim).days

cases_obs = cov19.filter_one_country(
    confirmed_cases, country, date_data_begin, date_data_end
)

# ------------------------------------------------------------------------------ #
# define change points for the three models
# ------------------------------------------------------------------------------ #

change_points_A = [
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
    dict(  # back to lambda 2 around easter
        pr_mean_date_begin_transient=datetime.datetime(2020, 4, 12),
        pr_sigma_date_begin_transient=1,
        pr_median_lambda=1 / 8,
        pr_sigma_lambda=0.2,
    ),
]

# add another change point in scenario B
change_points_B = copy.deepcopy(change_points_A)
change_points_B.append(
    dict(  # shortly after easter, people stop visiting and we might go back to lambda 3
        pr_mean_date_begin_transient=datetime.datetime(2020, 4, 14),
        pr_sigma_date_begin_transient=1,
        pr_median_lambda=1 / 8 / 2,
        pr_sigma_lambda=0.2,
    ),
)

# add another change point in scenario C
change_points_C = copy.deepcopy(change_points_B)
change_points_C.append(
    dict(  # German government will relax restrictions on April 19
        pr_mean_date_begin_transient=datetime.datetime(2020, 4, 19),
        pr_sigma_date_begin_transient=1,
        pr_median_lambda=1 / 8,
        pr_sigma_lambda=0.2,
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
)
model_B = cov19.SIR_with_change_points(
    np.diff(cases_obs),
    change_points_B,
    date_begin_sim,
    num_days_sim,
    diff_data_sim,
    N=83e6,
)
model_C = cov19.SIR_with_change_points(
    np.diff(cases_obs),
    change_points_C,
    date_begin_sim,
    num_days_sim,
    diff_data_sim,
    N=83e6,
)

trace_A = pm.sample(model=model_A, init="advi", cores=6)
print("Finished simulations for model A")
trace_B = pm.sample(model=model_B, init="advi", cores=6)
print("Finished simulations for model B")
trace_C = pm.sample(model=model_C, init="advi", cores=6)
print("Finished simulations for model C")

# ------------------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------------------ #

# set save path for figures relative to this script file
save_path = os.path.dirname(__file__) + "../../figures/easter_"
create_figure_timeseries(
    trace_A,
    color="tab:red",
    num_days_futu_to_plot=num_days_future,
    save_to=save_path + "ts_A",
)
create_figure_timeseries(
    trace_B,
    color="'tab:green'",
    num_days_futu_to_plot=num_days_future,
    save_to=save_path + "ts_B",
)
create_figure_timeseries(
    trace_C,
    color="'tab:orange'",
    num_days_futu_to_plot=num_days_future,
    save_to=save_path + "ts_C",
)

create_figure_distributions(
    model_A,
    trace_A,
    color="'tab:red'",
    num_changepoints=len(change_points_A),
    save_to=save_path + "dist_A",
)
create_figure_distributions(
    model_B,
    trace_B,
    color="'tab:green'",
    num_changepoints=len(change_points_B),
    save_to=save_path + "dist_B",
)
create_figure_distributions(
    model_C,
    trace_C,
    color="'tab:orange'",
    num_changepoints=len(change_points_C),
    save_to=save_path + "dist_C",
)

# ------------------------------------------------------------------------------ #
# Plotting helpers
# ------------------------------------------------------------------------------ #


def create_figure_timeseries(
    trace,
    color="tab:green",
    save_to=None,
    num_days_futu_to_plot=18,
    y_lim_lambda=(-0.15, 0.45),
    plot_red_axis=True,
):
    ylabel_new = f"New daily confirmed\ncases in {country}"
    ylabel_cum = f"Total confirmed\ncases in {country}"
    ylabel_lam = f"Effective\ngrowth rate $\lambda^\\ast (t)$"

    pos_letter = (-0.25, 1)
    titlesize = 16
    insetsize = ("25%", "50%")
    figsize = (4, 6)
    leg_loc = "upper right"

    new_c_ylim = [0, 15000]
    new_c_insetylim = [50, 17_000]

    cum_c_ylim = [0, 200_000]
    cum_c_insetylim = [50, 250_000]

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
    fig, axes = plt.subplots(
        3,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [2, 3, 3]},
        constrained_layout=True,
    )

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
        color=color,
        linewidth=2,
    )
    ax.fill_between(
        np.concatenate([mpl_dates_past[1:], mpl_dates_futu[1:]]),
        np.percentile(lambda_t - mu, q=2.5, axis=0),
        np.percentile(lambda_t - mu, q=97.5, axis=0),
        alpha=0.15,
        color=color,
        lw=0,
    )
    ax.set_ylabel(ylabel_lam)
    ax.set_ylim(*y_lim_lambda)
    ax.hlines(0, start_date, end_date, linestyles=":")
    delay = matplotlib.dates.date2num(date_data_end) - np.percentile(trace.delay, q=75)
    if plot_red_axis:
        ax.vlines(delay, -10, 10, linestyles="-", colors=["tab:red"])
        ax.text(
            delay + 1.5,
            0.4,
            "unconstrained due\nto reporting delay",
            color="tab:red",
            verticalalignment="top",
        )
        ax.text(
            delay - 1.5,
            0.4,
            "constrained\nby data",
            color="tab:red",
            horizontalalignment="right",
            verticalalignment="top",
        )
    ax.text(pos_letter[0], pos_letter[1], "A", transform=ax.transAxes, size=titlesize)
    ax.set_xlim(start_date, end_date)
    format_date_xticks(ax)
    for label in ax.xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)

    # --------------------------------------------------------------------------- #
    # New cases, lin scale first
    # --------------------------------------------------------------------------- #
    ax = axes[1]
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
        color=color,
        linewidth=1.5,
        label="Fit",
        zorder=10,
    )
    ax.fill_between(
        mpl_dates_past[1:],
        np.percentile(new_c_past, q=2.5, axis=0),
        np.percentile(new_c_past, q=97.5, axis=0),
        alpha=0.1,
        color=color,
        lw=0,
    )
    ax.plot(
        mpl_dates_futu[1:],
        np.median(new_c_futu, axis=0),
        "--",
        color=color,
        linewidth=3,
        label="Forecast",
    )
    ax.fill_between(
        mpl_dates_futu[1:],
        np.percentile(new_c_futu, q=2.5, axis=0),
        np.percentile(new_c_futu, q=97.5, axis=0),
        alpha=0.1,
        color=color,
        lw=0,
    )
    ax.fill_between(
        mpl_dates_futu[1:],
        np.percentile(new_c_futu, q=12.5, axis=0),
        np.percentile(new_c_futu, q=87.5, axis=0),
        alpha=0.2,
        color=color,
        lw=0,
    )
    ax.set_ylabel(ylabel_new)
    ax.legend(loc=leg_loc)
    ax.get_legend().get_frame().set_linewidth(0.0)
    ax.get_legend().get_frame().set_facecolor("#F0F0F0")
    ax.set_ylim(new_c_ylim)
    ax.set_xlim(start_date, end_date)
    ax.text(pos_letter[0], pos_letter[1], "B", transform=ax.transAxes, size=titlesize)

    ax.set_xlim(start_date, end_date)
    format_date_xticks(ax)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_k))
    for label in ax.xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)

    # NEW CASES LOG SCALE, skip forecast
    ax = inset_axes(ax, width=insetsize[0], height=insetsize[1], loc=2, borderpad=0.75)
    ax.plot(mpl_dates_past[1:], new_c_obsd, "d", markersize=2, label="Data", zorder=5)
    ax.plot(
        mpl_dates_past[1:],
        np.median(new_c_past, axis=0),
        "-",
        color=color,
        label="Fit (with 95% CI)",
        zorder=10,
    )
    ax.fill_between(
        mpl_dates_past[1:],
        np.percentile(new_c_past, q=2.5, axis=0),
        np.percentile(new_c_past, q=97.5, axis=0),
        alpha=0.1,
        color=color,
        lw=0,
    )
    format_date_xticks(ax, minor=True)
    ax.set_yscale("log")
    ax.set_yticks([1e1, 1e2, 1e3, 1e4, 1e5])
    ax.set_xlim(start_date, mid_date)
    ax.set_ylim(new_c_insetylim)
    ax.yaxis.tick_right()
    for label in ax.xaxis.get_ticklabels()[1:-1]:
        label.set_visible(False)

    # --------------------------------------------------------------------------- #
    # Total cases, lin scale first
    # --------------------------------------------------------------------------- #
    ax = axes[2]
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
        color=color,
        linewidth=1.5,
        label="Fit with 95% CI",
        zorder=10,
    )
    ax.fill_between(
        mpl_dates_past[:],
        np.percentile(cum_c_past, q=2.5, axis=0),
        np.percentile(cum_c_past, q=97.5, axis=0),
        alpha=0.1,
        color=color,
        lw=0,
    )
    ax.plot(
        mpl_dates_futu[1:],
        np.median(cum_c_futu[:, 1:], axis=0),
        "--",
        color=color,
        linewidth=3,
        label="Forecast with 75% and 95% CI",
    )
    ax.fill_between(
        mpl_dates_futu[1:],
        np.percentile(cum_c_futu[:, 1:], q=2.5, axis=0),
        np.percentile(cum_c_futu[:, 1:], q=97.5, axis=0),
        alpha=0.1,
        color=color,
        lw=0,
    )
    ax.fill_between(
        mpl_dates_futu[1:],
        np.percentile(cum_c_futu[:, 1:], q=12.5, axis=0),
        np.percentile(cum_c_futu[:, 1:], q=87.5, axis=0),
        alpha=0.2,
        color=color,
        lw=0,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel_cum)
    ax.set_ylim(cum_c_ylim)
    ax.set_xlim(start_date, end_date)
    ax.text(pos_letter[0], pos_letter[1], "C", transform=ax.transAxes, size=titlesize)

    ax.set_xlim(start_date, end_date)
    format_date_xticks(ax)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_k))
    for label in ax.xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)

    # Total CASES LOG SCALE, skip forecast
    ax = inset_axes(ax, width=insetsize[0], height=insetsize[1], loc=2, borderpad=0.75)
    ax.plot(mpl_dates_past[:], cum_c_obsd, "d", markersize=2, label="Data", zorder=5)
    ax.plot(
        mpl_dates_past[:],
        np.median(cum_c_past, axis=0),
        "-",
        color=color,
        label="Fit (with 95% CI)",
        zorder=10,
    )
    ax.fill_between(
        mpl_dates_past[:],
        np.percentile(cum_c_past, q=2.5, axis=0),
        np.percentile(cum_c_past, q=97.5, axis=0),
        alpha=0.1,
        color=color,
        lw=0,
    )
    format_date_xticks(ax, minor=True)
    ax.set_yscale("log")
    ax.set_yticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    ax.set_xlim(start_date, mid_date)
    ax.set_ylim(cum_c_insetylim)
    ax.yaxis.tick_right()
    for label in ax.xaxis.get_ticklabels()[1:-1]:
        label.set_visible(False)

    # --------------------------------------------------------------------------- #
    # Finalize
    # --------------------------------------------------------------------------- #

    for ax in axes:
        ax.set_rasterization_zorder(rasterization_zorder)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    # plt.subplots_adjust(wspace=0.4, hspace=0.25)
    if save_to is not None:
        plt.savefig(
            save_to + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0,
        )
        plt.savefig(
            save_to + ".png", dpi=300, bbox_inches="tight", pad_inches=0,
        )


def get_label_dict():
    labels = dict()
    labels["mu"] = f"Recovery rate"
    labels["delay"] = f"Reporting delay"
    labels["I_begin"] = f"Initial infections"
    labels["sigma_obs"] = f"Scale (width)\nof the likelihood"
    labels["lambda_0"] = f"Initial rate"
    labels["lambda_1"] = f"Spreading rates"
    labels["lambda_2"] = f""
    labels["lambda_3"] = f""
    labels["lambda_4"] = f""
    labels["transient_begin_0"] = f"Change times"
    labels["transient_begin_1"] = f""
    labels["transient_begin_2"] = f""
    labels["transient_begin_3"] = f""
    labels["transient_len_0"] = f"Change duration"
    labels["transient_len_1"] = f""
    labels["transient_len_2"] = f""
    labels["transient_len_3"] = f""
    labels["E_begin_scale"] = "Initial scale\nof exposed"
    labels["median_incubation"] = "Median\nincubation delay"
    labels["sigma_random_walk"] = "Std. of\nrandom walk"
    labels["weekend_factor"] = "Factor\nweekends discounted"
    labels["offset_modulation_rad"] = "Offset from sunday\nof the modulation"
    return labels


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
        ax.set_xlabel(labels[key])
        ax.xaxis.set_label_position("top")

        # make some bold
        if key == "lambda_1" or key == "transient_begin_0" or key == "transient_len_0":
            ax.set_xlabel(labels[key], fontweight="bold")

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

        if False:
            ax.text(
                0.05,
                0.9,
                text,
                horizontalalignment="center",
                verticalalignment="top",
                transform=ax.transAxes,
                bbox=dict(facecolor="white", alpha=0.3, edgecolor="none"),
                fontsize=12,
            )
        else:
            if key in insets.keys():
                # strip everything except the median value
                text = text.replace("Median: ", "").replace("CI: ", "")
                md = text.split("\n")[0]
                ci = text.split("\n")[1]

                # matplotlib.rcParams['text.usetex'] = True
                # with rc_context(rc={'text.usetex': True}):
                text = insets[key] + md + "$" + "\n" + r"$\,$"
                ax.text(
                    0.6,
                    0.9,
                    text,
                    fontsize=12,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    horizontalalignment="center",
                    bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
                )
                ax.text(
                    0.6,
                    0.6,
                    ci,
                    fontsize=9,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    horizontalalignment="center",
                    # bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
                ),

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
# Formatting helpers
# ------------------------------------------------------------------------------ #

# format yaxis 10_000 as 10 k
format_k = lambda num, _: "${:.0f}\,$k".format(num / 1_000)

# format xaxis, ticks and labels
def format_date_xticks(ax, minor=True):
    ax.xaxis.set_major_locator(
        matplotlib.dates.WeekdayLocator(interval=1, byweekday=matplotlib.dates.SU)
    )
    if minor:
        ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%d"))


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
