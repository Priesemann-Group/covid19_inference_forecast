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

""" ## Load data 
"""
data_begin = datetime.datetime.now() - datetime.timedelta(days=7 * 12)
data_end = datetime.datetime.now() - datetime.timedelta(days=2)
rki = cov19.data_retrieval.RKI(True)
rki.download_all_available_data(force_download=False)
new_cases_obs = rki.get_new("confirmed", data_begin=data_begin, data_end=data_end)
total_cases_obs = rki.get_total("confirmed", data_begin=data_begin, data_end=data_end)


with open("./data/what_if_lockdown_jan.pickled", "rb") as f:
    [(mod_a, mod_b, mod_c), (tr_a, tr_b, tr_c)] = pickle.load(f)


try:
    # only works when called from python, not reliable in interactive ipython etc.
    os.chdir(os.path.dirname(__file__))
    save_to = "./figures/what_if_jan_"
except:
    # assume base directory
    save_to = "./figures/what_if_jan_"

""" ### Timeseries
    Timeseries overview, for now needs an offset variable to get cumulative cases
"""  #### German


cov19.plot.set_rcparams(cov19.plot.get_rcparams_default())
cov19.plot.rcParams.draw_ci_50 = False
cov19.plot.rcParams.draw_ci_75 = False
cov19.plot.rcParams.draw_ci_95 = True
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
    forecast_label="No changes in dynamic",
    color="#225ea8",
    forecast_heading=r"$\bf Scenarios\!:$",
    add_more_later=True,
)

fig, axes = create_plot_scenarios(
    mod_b,
    tr_b,
    axes=axes,
    offset=total_cases_obs[0],
    forecast_label=f"$R=1.3$",
    color="#41b6c4",
)

fig, axes = create_plot_scenarios(
    mod_a,
    tr_a,
    axes=axes,
    offset=total_cases_obs[0],
    forecast_label=f"R=0.7",
    color="tab:green",
)


# Set lambda labels and limit
axes[0].set_ylim(-0.12, 0.22)
axes[0].set_ylabel("Effective\ngrowth rate")


# Set new cases limit and labels
axes[1].set_ylabel("New cases per\n 1.000.000 inhabitants")
axes[1].set_xlabel("Date")
ylim = (0,2000)
axes[1].set_ylim(ylim)

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
    date_ld - datetime.timedelta(days=9),
    ls=":",
    color="tab:gray",
    zorder=0,
)
axes[1].axvline(date_ld, ls=":", color="tab:gray", zorder=0)
axes[0].text(
    date_ld - datetime.timedelta(days=9, hours=20),
    0.25,
    "Inference",
    ha="right",
    color="tab:gray",
    size=8,
)
axes[0].text(
    date_ld - datetime.timedelta(days=8, hours=20),
    0.25,
    "Forecast",
    ha="left",
    color="tab:gray",
    size=8,
)
axes[1].text(
    date_ld - datetime.timedelta(hours=20),
    ylim[1]-30,
    "Inference",
    ha="right",
    color="tab:gray",
    size=8,
    zorder=0,
)
axes[1].text(
    date_ld + datetime.timedelta(hours=20),
    ylim[1]-30,
    "Forecast",
    ha="left",
    color="tab:gray",
    size=8,
    zorder=0,
)

# Annotations ld_date
"""axes[0].axvline(
    datetime.datetime(2020, 12, 14), ls="-.", color="#99bbff", zorder=0,
)
axes[1].axvline(
    datetime.datetime(2020, 12, 14), ls="-.", color="#99bbff", zorder=0,
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
            "confirmed",
            data_begin=data_begin,
            data_end=datetime.datetime.now(),
        )
        / 83.02e6
        * 1e6
    )
    .rolling(7)
    .mean()
)
cov19.plot._timeseries(
    x=new_cases_obs.index,
    y=new_cases_obs,
    ax=axes[1],
    what="data",
    zorder=0,
)


# Set limit for x axes
for ax in axes:
    ax.set_xlim(
        data_end - datetime.timedelta(weeks=8), data_end + datetime.timedelta(weeks=8)
    )
    # Lets try this
    locator = mdates.MonthLocator()
    formatter = mdates.DateFormatter("%d. %b")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

# Remove .0 from yaxis
axes[1].get_yaxis().set_major_formatter(
    mpl.ticker.FuncFormatter(lambda x, p: format(int(x)))
)


""" Add current R RKI
"""


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


last = 0
for varname in tr_c.varnames:
    if varname.startswith("transient_day_"):
        length = len(varname.split("_"))
        if length == 3:
            _, _, num = varname.split("_")
            if int(num) > last:
                last = int(num)

f_trunc = lambda number, precision: "{{:.{}f}}".format(precision).format(number)

current_R = (tr_c[f"lambda_{last}"] - tr_c["mu"] + 1) ** 4
from covid19_inference_new.plot import (
    _get_mpl_text_coordinates,
    _add_mpl_rect_around_text,
)

med, perc1, perc2 = mean_confidence_interval(current_R)
text_md = f"{f_trunc(med,3)}"
text_ci = f"[{f_trunc(perc1,3)}, {f_trunc(perc2,3)}]"
tel_md = axes[0].text(
    0.2,
    0.95,  # let's have a ten percent margin or so
    r"Current $R_{RKI} \simeq " + text_md + r"$",
    fontsize=10,
    transform=axes[0].transAxes,
    verticalalignment="top",
    horizontalalignment="center",
    zorder=100,
)
x_min, x_max, y_min, y_max = _get_mpl_text_coordinates(tel_md, axes[0])
"""tel_ci = axes[0].text(
    0.8,
    y_min * 0.9,  # let's have a ten percent margin or so
    text_ci,
    fontsize=8,
    transform=axes[0].transAxes,
    verticalalignment="top",
    horizontalalignment="center",
    zorder=101,
)
"""
_add_mpl_rect_around_text(
    [tel_md],
    axes[0],
    facecolor="#F0F0F0",
    alpha=0.5,
    zorder=99,
)


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
