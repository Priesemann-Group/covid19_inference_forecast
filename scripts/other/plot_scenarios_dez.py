# ------------------------------------------------------------------------------ #
# Run to plot new cases onto the old data, run what_if_lockdown_nov to generate
# the corresponding data!
# Runtime: 1min
# ------------------------------------------------------------------------------ #
import datetime
import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
import theano
import theano.tensor as tt
import pymc3 as pm
import pickle
import matplotlib.dates as mdates

try:
    import covid19_inference_new as cov19
except ModuleNotFoundError:
    sys.path.append("../..")
    import covid19_inference_new as cov19
import logging

log = logging.getLogger(__name__)

## Plotting file
from plot import create_plot_scenarios

""" ## Load data (revise once we reach Dez 14th)
"""
data_begin = datetime.datetime(2020, 8, 1)
data_end = datetime.datetime(2020, 12, 14)
rki = cov19.data_retrieval.RKI(True)
rki.download_all_available_data(force_download=False)
new_cases_obs = rki.get_new("confirmed", data_begin=data_begin, data_end=data_end)
total_cases_obs = rki.get_total("confirmed", data_begin=data_begin, data_end=data_end)


with open("./data/what_if_lockdown_dez.pickled", "rb") as f:
    [(mod_a, mod_b, mod_c), (tr_a, tr_b, tr_c)] = pickle.load(f)


try:
    # only works when called from python, not reliable in interactive ipython etc.
    os.chdir(os.path.dirname(__file__))
    save_to = "./figures/what_if_lockdown_"
except:
    # assume base directory
    save_to = "./figures/what_if_lockdown_"

""" ### Timeseries
    Timeseries overview, for now needs an offset variable to get cumulative cases
"""  #### German


cov19.plot.set_rcparams(cov19.plot.get_rcparams_default())
cov19.plot.rcParams.draw_ci_50 = False
cov19.plot.rcParams.draw_ci_75 = True
cov19.plot.rcParams.draw_ci_95 = False
# cov19.plot.rcParams.locale = "de_de"
cov19.plot.rcParams.date_format = "%d. %b"
cov19.plot.rcParams.fcast_ls = "-"

mpl.rcParams["font.sans-serif"] = "Arial"
mpl.rcParams["font.family"] = "sans-serif"

# Create plots
fig, axes = create_plot_scenarios(
    mod_c,
    tr_c,
    offset=total_cases_obs[0],
    forecast_label="No contact reduction",
    color="#225ea8",
    forecast_heading=r"$\bf Scenarios\!:$",
    add_more_later=True,
)

fig, axes = create_plot_scenarios(  # Strenger 2.nov
    mod_b,
    tr_b,
    axes=axes,
    offset=total_cases_obs[0],
    forecast_label=f"Mild contact reduction starting Dec 14th",
    color="#41b6c4",
)

fig, axes = create_plot_scenarios(
    mod_a,
    tr_a,
    axes=axes,
    offset=total_cases_obs[0],
    forecast_label=f"Strong contact reduction starting Dec 14th",
    color="tab:green",
)

# Set lambda labels and limit
axes[0].set_ylim(-0.15, 0.15)
axes[0].set_ylabel("Effective\ngrowth rate")


# Set new cases limit and labels
axes[1].set_ylabel("New cases per\n 1.000.000 inhabitants")
axes[1].set_xlabel("Date")
axes[1].set_ylim(0, 300)

# Disable total cases axes visuals
axes[2].set_ylim(0, 0)
axes[2].get_xaxis().set_visible(False)
axes[2].get_yaxis().set_visible(False)
axes[2].spines["bottom"].set_visible(False)
axes[2].spines["left"].set_visible(False)
axes[2].texts[0].set_visible(False)  # Remove C letter


# R lines
axes[0].axhline((1.3) ** (1 / 4) - 1.0, ls=":", color="#000000", zorder=0)
axes[0].axhline((0.7) ** (1 / 4) - 1.0, ls=":", color="#969696", zorder=0)


# Annotations forecast/prognose lines
date_ld = new_cases_obs.index[-1]
axes[0].axvline(
    date_ld - datetime.timedelta(days=9), ls=":", color="tab:gray", zorder=0,
)
axes[1].axvline(date_ld, ls=":", color="tab:gray", zorder=0)
axes[0].text(
    date_ld - datetime.timedelta(days=9, hours=12),
    0.12,
    "Inference",
    ha="right",
    color="tab:gray",
    size=8,
)
axes[0].text(
    date_ld - datetime.timedelta(days=8, hours=12),
    0.12,
    "Forecast",
    ha="left",
    color="tab:gray",
    size=8,
)
axes[1].text(
    date_ld - datetime.timedelta(hours=12),
    290,
    "Inference",
    ha="right",
    color="tab:gray",
    size=8,
)
axes[1].text(
    date_ld + datetime.timedelta(hours=12),
    290,
    "Forecast",
    ha="left",
    color="tab:gray",
    size=8,
)

# Annotations ld_date
axes[0].axvline(
    datetime.datetime(2020, 12, 14), ls="-.", color="#99bbff", zorder=0,
)
axes[1].axvline(
    datetime.datetime(2020, 12, 14), ls="-.", color="#99bbff", zorder=0,
)


# Annotations first ld lines
"""
date_first_ld = datetime.datetime(2020, 11, 2)
axes[0].axvline(date_first_ld, ls=":", color="tab:orange", zorder=0)
axes[1].axvline(
    date_first_ld + datetime.timedelta(days=9), ls=":", color="tab:orange", zorder=0,
)
axes[0].text(
    date_first_ld - datetime.timedelta(hours=12),
    0.12,
    "Ld light",
    ha="right",
    color="tab:orange",
    size=8,
)
axes[1].text(
    date_first_ld + datetime.timedelta(days=8, hours=12),
    490,
    "Ld light",
    ha="right",
    color="tab:orange",
    size=8,
)
"""


legend = axes[2].get_legend()
legend._loc = 10  # center legend
legend.get_texts()[0].set_text("Data (RKI Meldedatum) smoothed")  # Add to Data legend

# Change size of plot
fig.set_size_inches(5, 5)
# Set ratios
fig._gridspecs[0].set_height_ratios([1, 3.5, 1.5])


# Plot total new cases
new_cases_obs = (
    (
        rki.get_new(
            "confirmed", data_begin=data_begin, data_end=datetime.datetime.now(),
        )
        / 83.02e6
        * 1e6
    )
    .rolling(7)
    .mean()
)
cov19.plot._timeseries(
    x=new_cases_obs.index, y=new_cases_obs, ax=axes[1], what="data", zorder=0,
)


# Set limit for x axes
for ax in axes:
    ax.set_xlim(datetime.datetime(2020, 10, 15), datetime.datetime(2021, 1, 15))
    # Lets try this
    locator = mdates.MonthLocator()
    formatter = mdates.DateFormatter("%d. %b")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


fig.savefig(
    save_to + "english_ts.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
    transparent=True,
)
fig.savefig(
    save_to + "english_ts.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
    transparent=True,
)

"""### Distributions

#Do for each of the 3 models


for this_model, trace, change_points, color, name in zip(
    [mod_a, mod_b, mod_c],
    [tr_a, tr_b, tr_c],
    [cp_a, cp_b, cp_c],
    ["tab:red", "tab:orange", "tab:green"],
    ["lockdown_1", "lockdown_2", "lockdown_3"],
):

    num_rows = len(change_points) + 1 + 1
    num_columns = 3
    fig_width = 4.5 / 3 * num_columns
    fig_height = num_rows * 1

    fig, axes = plt.subplots(
        num_rows, num_columns, figsize=(fig_width, fig_height), constrained_layout=True
    )
    # Left row we want mu and all lambda_i!
    for i in range(num_rows):
        if i == 0:
            cov19.plot._distribution(this_model, trace, "mu", axes[0, 0], color=color)
        elif i == 1:
            # Plot lambda_i and remove the xlable, we add one big label later.
            cov19.plot._distribution(
                this_model, trace, f"lambda_{i-1}", axes[i, 0], color=color
            )
            axes[i, 0].set_xlabel("Inital rate")
        else:
            # Plot lambda_i and remove the xlable, we add one big label later.
            cov19.plot._distribution(
                this_model, trace, f"lambda_{i-1}", axes[i, 0], color=color
            )
            axes[i, 0].set_xlabel("")
    # middle row
    for i in range(num_rows):
        if i == 0:
            cov19.plot._distribution(
                this_model, trace, "sigma_obs", axes[i, 1], color=color
            )
        elif i == 1:
            cov19.plot._distribution(
                this_model, trace, "I_begin", axes[i, 1], color=color
            )
        else:
            # Plot transient_day_i and remove the xlable, we add one big label later.
            cov19.plot._distribution(
                this_model, trace, f"transient_day_{i-1}", axes[i, 1], color=color
            )
            axes[i, 1].set_xlabel("")
    # right row
    for i in range(num_rows):
        if i == 0:
            # Create legend for everything
            axes[i, 2].set_axis_off()
            axes[i, 2].plot(
                [],
                [],
                color=cov19.plot.rcParams["color_prior"],
                linewidth=3,
                label="Prior",
            )
            axes[i, 2].hist([], color=color, label="Posterior")
            axes[i, 2].legend(loc="center left")
            axes[i, 2].get_legend().get_frame().set_linewidth(0.0)
            axes[i, 2].get_legend().get_frame().set_facecolor("#F0F0F0")
        elif i == 1:
            cov19.plot._distribution(
                this_model, trace, f"delay", axes[i, 2], color=color
            )
            axes[i, 2].set_xlabel("Reporting delay")
        else:
            # Plot transient_len_i and remove the xlable, we add one big label later.
            cov19.plot._distribution(
                this_model, trace, f"transient_len_{i-1}", axes[i, 2], color=color
            )
            axes[i, 2].set_xlabel("")

    # Add ylabel for the first axes
    axes[0, 0].set_ylabel("Density")
    # Set bold xlabel for Spreading rates Change times and Change durations
    axes[2, 0].set_xlabel("Spreading rates", fontweight="bold")
    axes[2, 1].set_xlabel("Change times", fontweight="bold")
    axes[2, 2].set_xlabel("Change duration", fontweight="bold")

    # Letters
    letter_kwargs = dict(x=-0.3, y=1.1, size="x-large")
    axes[0, 0].text(s="D", transform=axes[0, 0].transAxes, **letter_kwargs)
    axes[1, 0].text(s="E", transform=axes[1, 0].transAxes, **letter_kwargs)
    axes[2, 0].text(s="F", transform=axes[2, 0].transAxes, **letter_kwargs)

    # Save to file
    fig.savefig(
        save_to + "dist_" + name + ".pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
    )
    fig.savefig(
        save_to + "dist_" + name + ".png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
    )

"""
