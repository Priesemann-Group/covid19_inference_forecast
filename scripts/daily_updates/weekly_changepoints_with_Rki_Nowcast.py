"""
    # Example for weekly changepoints

    Runtime ~ 1h

    ## Importing modules
"""

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

try:
    import covid19_inference_new as cov19
except ModuleNotFoundError:
    sys.path.append("../..")
    import covid19_inference_new as cov19


""" ## Data retrieval
"""

# Get nowcasting data: they have a github repo now... wow
url = "https://raw.githubusercontent.com/robert-koch-institut/SARS-CoV-2-Nowcasting_und_-R-Schaetzung/main/Nowcast_R_aktuell.csv"
df = pd.read_csv(url)
df["date"] = pd.to_datetime(df["Datum"], format="%Y-%m-%d")
df = df.set_index(df["date"])

df["new_cases"] = df["PS_COVID_Faelle"]
df["total_cases"] = df["new_cases"].cumsum(axis=0)

data_begin = datetime.datetime.today() - datetime.timedelta(days=30 * 3)
data_end = df["new_cases"].index[-1]


new_cases_obs = df["new_cases"]
total_cases_obs = df["total_cases"]
df = df[data_begin : (data_end - datetime.timedelta(days=1))]
df["Punktschätzer des 7-Tage-R Wertes"] = (
    df["PS_7_Tage_R_Wert"].astype(str).str.replace(",", ".").astype(float)
)
df["Untere Grenze des 95%-Prädiktionsintervalls des 7-Tage-R Wertes"] = (
    df["PS_7_Tage_R_Wert"].astype(str).str.replace(",", ".").astype(float)
)
df["Obere Grenze des 95%-Prädiktionsintervalls des 7-Tage-R Wertes"] = (
    df["OG_PI_7_Tage_R_Wert"].astype(str).str.replace(",", ".").astype(float)
)

## Additionally we are loading the normal data to plot the last 4 days without inputation

rki = cov19.data_retrieval.RKI(False)
rki.download_all_available_data(force_download=True)
new_cases_obs_raw = rki.get_new(
    value="confirmed", data_begin=data_end, data_end=datetime.datetime.today()
)


""" ## Create weekly changepoints

    TODO:
        Get relative_to_previous working
"""

# Change point midpoint dates

# Structures change points in a dict. Variables not passed will assume default values.
change_points = [
    dict(
        pr_mean_date_transient=data_begin - datetime.timedelta(days=1),
        pr_sigma_date_transient=1.5,
        pr_median_lambda=0.12,
        pr_sigma_lambda=0.5,
        pr_sigma_transient_len=0.5,
    ),
]
print(f"Adding possible change points at:")
for i, day in enumerate(pd.date_range(start=data_begin, end=datetime.datetime.now())):
    if day.weekday() == 6 and (datetime.datetime.today() - day).days > 9:
        print(f"\t{day}")

        # Prior factor to previous
        change_points.append(
            dict(  # one possible change point every sunday
                pr_mean_date_transient=day,
                pr_sigma_date_transient=1.5,
                pr_sigma_lambda=0.2,  # wiggle compared to previous point
                relative_to_previous=True,
                pr_factor_to_previous=1,
            )
        )

""" ## Put the model together
"""

# Number of days the simulation starts earlier than the data.
# Should be significantly larger than the expected delay in order to always fit the same number of data points.
diff_data_sim = 16
# Number of days in the future (after date_end_data) to forecast cases
num_days_forecast = 10
params_model = dict(
    new_cases_obs=new_cases_obs[data_begin:],
    data_begin=data_begin,
    fcast_len=num_days_forecast,
    diff_data_sim=diff_data_sim,
    N_population=83e6,
)
# Median of the prior for the delay in case reporting, we assume 10 days
pr_delay = 4
with cov19.model.Cov19Model(**params_model) as this_model:
    # Create the an array of the time dependent infection rate lambda
    lambda_t_log = cov19.model.lambda_t_with_sigmoids(
        pr_median_lambda_0=0.4,
        pr_sigma_lambda_0=0.5,
        change_points_list=change_points,  # The change point priors we constructed earlier
        name_lambda_t="lambda_t",  # Name for the variable in the trace (see later)
    )

    # set prior distribution for the recovery rate
    mu = pm.Lognormal(name="mu", mu=np.log(1 / 8), sigma=0.2)

    # This builds a decorrelated prior for I_begin for faster inference.
    # It is not necessary to use it, one can simply remove it and use the default argument
    # for pr_I_begin in cov19.SIR
    prior_I = cov19.model.uncorrelated_prior_I(
        lambda_t_log=lambda_t_log,
        mu=mu,
        pr_median_delay=pr_delay,
        name_I_begin="I_begin",
        name_I_begin_ratio_log="I_begin_ratio_log",
        pr_sigma_I_begin=2,
        n_data_points_used=5,
    )

    # Use lambda_t_log and mu to run the SIR model
    new_cases = cov19.model.SIR(
        lambda_t_log=lambda_t_log,
        mu=mu,
        name_new_I_t="new_I_t",
        name_I_t="I_t",
        name_I_begin="I_begin",
        pr_I_begin=prior_I,
    )

    # Delay the cases by a lognormal reporting delay
    new_cases = cov19.model.delay_cases(
        cases=new_cases,
        name_cases="delayed_cases",
        name_delay="delay",
        name_width="delay-width",
        pr_mean_of_median=pr_delay,
        pr_sigma_of_median=0.2,
        pr_median_of_width=0.3,
    )

    # Modulate the inferred cases by a abs(sin(x)) function, to account for weekend effects
    # Also adds the "new_cases" variable to the trace that has all model features.
    new_cases = cov19.model.week_modulation(
        cases=new_cases,
        name_cases="new_cases",
        name_weekend_factor="weekend_factor",
        name_offset_modulation="offset_modulation",
        week_modulation_type="abs_sine",
        pr_mean_weekend_factor=0.3,
        pr_sigma_weekend_factor=0.5,
        weekend_days=(6, 7),
    )

    # Define the likelihood, uses the new_cases_obs set as model parameter
    cov19.model.student_t_likelihood(new_cases)

""" ## MCMC sampling
"""

trace = pm.sample(model=this_model, init="advi", tune=1000, draws=500)


""" ## Plotting
    
    ### Save path
"""
try:
    # only works when called from python, not reliable in interactive ipython etc.
    os.chdir(os.path.dirname(__file__))
    save_to = "../../figures/weekly_cps_nowcast_"
except:
    # assume base directory
    save_to = "../../figures/weekly_cps_nowcast_"

""" ### Timeseries
    Timeseries overview, for now needs an offset variable to get cumulative cases
"""
cov19.plot.rcParams["color_model"] = "tab:orange"
fig, axes = cov19.plot.timeseries_overview(this_model, trace, offset=total_cases_obs[0])

for ax in axes:
    ax.set_xlim(datetime.datetime.now() - datetime.timedelta(days=4 * 17))

# Set ylim for new cases
axes[1].set_ylim(0, new_cases_obs.max() + 10000)

# --------------------------------------------------------------------------- #
# inset new cases
# --------------------------------------------------------------------------- #
# Add inset for march to juli
"""
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

axins = axes[1].inset_axes(bounds=[0.1, 0.5, 0.4, 0.4])
for line in axes[1].lines:
    axins.lines.append(line)

ax = axins

# model fit
cov19.plot._timeseries(
    x=new_cases_obs.index,
    y=new_cases_obs,
    ax=ax,
    what="model",
    color="tab:orange",
)
prec = 1.0 / (np.log10(ax.get_ylim()[1]) - 2.5)
if prec < 2.0 and prec >= 0:
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(cov19.plot._format_k(int(prec)))
    )
    ticks = ax.get_xticks()
    ax.set_xticks(ticks=[new_cases_obs.index.min(), new_cases_obs.index.max()])
"""

# Set y lim for effective growth rate
axes[0].set_ylim(-0.08, 0.1)

mu = np.mean(trace["mu"])
cov19.plot._timeseries(
    x=df["Punktschätzer des 7-Tage-R Wertes"].dropna().index,
    y=df["Punktschätzer des 7-Tage-R Wertes"].dropna() ** 0.25 - 1,  # R*mu=lambda
    ax=axes[0],
    color="tab:purple",
    label=r"$\sqrt[4]{R_{RKI}}-1$",
    what="model",
    ls="--",
)

# Plot 95% ci
axes[0].fill_between(
    df["Untere Grenze des 95%-Prädiktionsintervalls des 7-Tage-R Wertes"]
    .dropna()
    .index,
    df["Untere Grenze des 95%-Prädiktionsintervalls des 7-Tage-R Wertes"].dropna()
    ** 0.25
    - 1,
    df["Obere Grenze des 95%-Prädiktionsintervalls des 7-Tage-R Wertes"].dropna()
    ** 0.25
    - 1,
    alpha=0.1,
    lw=0,
    color="tab:purple",
)


# Plot the raw data
cov19.plot._timeseries(
    x=new_cases_obs_raw.index,
    y=new_cases_obs_raw,
    ax=axes[1],
    color="tab:red",
    label=r"Data (no imputation)",
    what="data",
)
axes[1].legend()
handles, labels = axes[1].get_legend_handles_labels()
order = [0, 2, 1]
axes[1].legend(
    [handles[idx] for idx in order], [labels[idx] for idx in order], loc="lower left"
)
axes[1].get_legend().get_frame().set_linewidth(0.0)
axes[1].get_legend().get_frame().set_facecolor("#F0F0F0")

axes[0].legend(loc="upper left")
axes[0].get_legend().get_frame().set_linewidth(0.0)
axes[0].get_legend().get_frame().set_facecolor("#F0F0F0")

""" Add text for current reproduction number
"""


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


f_trunc = lambda number, precision: "{{:.{}f}}".format(precision).format(number)
num_rows = len(change_points) + 1 + 1
last = num_rows - 2
current_R = (trace[f"lambda_{last}"] - trace["mu"] + 1) ** 4
from covid19_inference_new.plot import (
    _get_mpl_text_coordinates,
    _add_mpl_rect_around_text,
)

med, perc1, perc2 = mean_confidence_interval(current_R)
text_md = f"{f_trunc(med,3)}"
text_ci = f"[{f_trunc(perc1,3)}, {f_trunc(perc2,3)}]"
tel_md = axes[0].text(
    0.8,
    0.9,  # let's have a ten percent margin or so
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

# Add vline for today
# axes[0].axvline(datetime.datetime.today(), ls=":", color="tab:gray")
# axes[1].axvline(datetime.datetime.today(), ls=":", color="tab:gray")
# axes[2].axvline(datetime.datetime.today(), ls=":", color="tab:gray")
# ts for timeseries
plt.savefig(
    save_to + "ts.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.savefig(
    save_to + "ts.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
""" ### Distributions
"""
num_rows = len(change_points) + 1 + 1
num_columns = int(np.ceil(14 / 5))
fig_width = 4.5 / 3 * num_columns
fig_height = 8 * 1

fig, axes = plt.subplots(
    8, num_columns, figsize=(fig_width, fig_height), constrained_layout=True
)
# Left row we want mu and all lambda_i!

rows = [
    0,
    num_rows,
    num_rows - 1,
    num_rows - 2,
    num_rows - 3,
    num_rows - 4,
    num_rows - 5,
    num_rows - 6,
]
for i in rows:
    if i == 0:
        cov19.plot._distribution(this_model, trace, "mu", axes[0, 0])
    elif i == num_rows:
        # Plot lambda_i and remove the xlable, we add one big label later.
        cov19.plot._distribution(this_model, trace, f"lambda_{0}", axes[1, 0])
        axes[1, 0].set_xlabel("Inital rate")
    else:
        # Plot lambda_i and remove the xlable, we add one big label later.
        cov19.plot._distribution(
            this_model, trace, f"lambda_{i-2}", axes[-i + num_rows + 1, 0]
        )
        axes[-i + num_rows + 1, 0].set_xlabel("")
# middle row
for i in rows:
    if i == 0:
        cov19.plot._distribution(this_model, trace, "sigma_obs", axes[i, 1])
    elif i == num_rows:
        cov19.plot._distribution(this_model, trace, "I_begin", axes[1, 1])
    else:
        # Plot transient_day_i and remove the xlable, we add one big label later.
        cov19.plot._distribution(
            this_model, trace, f"transient_day_{i-2}", axes[-i + num_rows + 1, 1]
        )
        axes[-i + num_rows + 1, 1].set_xlabel("")
# right row
for i in rows:
    if i == 0:
        # Create legend for everything
        axes[i, 2].set_axis_off()
        axes[i, 2].plot(
            [], [], color=cov19.plot.rcParams["color_prior"], linewidth=3, label="Prior"
        )
        axes[i, 2].hist([], color=cov19.plot.rcParams["color_model"], label="Posterior")
        axes[i, 2].legend(loc="center left")
        axes[i, 2].get_legend().get_frame().set_linewidth(0.0)
        axes[i, 2].get_legend().get_frame().set_facecolor("#F0F0F0")
    elif i == num_rows:
        cov19.plot._distribution(this_model, trace, f"delay", axes[1, 2])
        axes[1, 2].set_xlabel("Reporting delay")
    else:
        # Plot transient_len_i and remove the xlable, we add one big label later.
        cov19.plot._distribution(
            this_model, trace, f"transient_len_{i-2}", axes[-i + num_rows + 1, 2]
        )
        axes[-i + num_rows + 1, 2].set_xlabel("")
    if i == 2:
        axes[i, 2].set_xlim(18, 40)

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

# dist for distributions
plt.savefig(
    save_to + "dist.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
    transparent=True,
)
plt.savefig(
    save_to + "dist.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
    transparent=True,
)
