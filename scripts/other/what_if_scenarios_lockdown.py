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
import logging

log = logging.getLogger(__name__)

""" ## Data retrieval
"""
data_begin = datetime.datetime(2020, 9, 1)
data_end = datetime.datetime.now()
rki = cov19.data_retrieval.RKI(True)
rki.download_all_available_data(force_download=True)
new_cases_obs = rki.get_new("confirmed", data_begin=data_begin, data_end=data_end)
total_cases_obs = rki.get_total("confirmed", data_begin=data_begin, data_end=data_end)


""" ## Create weekly changepoints up to the 12. Oktober
"""
change_points = [
    dict(  # one possible change point every sunday
        pr_mean_date_transient=data_begin - datetime.timedelta(days=3),
        pr_sigma_date_transient=1.5,
        pr_median_lambda=0.12,
        pr_sigma_lambda=0.5,  # No wiggle
    )
]
log.info(f"Adding possible change points at:")
for i, day in enumerate(pd.date_range(start=data_begin, end=data_end)):
    if day.weekday() == 0 and day < datetime.datetime.now() - datetime.timedelta(
        days=10
    ):
        log.info(f"\t{day}")

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


""" ## Manual add last cps for scenarios
"""
import copy

cp_a = copy.copy(change_points)
cp_b = copy.copy(change_points)
cp_c = copy.copy(change_points)


cp_a.append(  # Lockdown start now
    dict(
        pr_mean_date_transient=datetime.datetime(2020, 11, 1)
        + datetime.timedelta(days=1),  # shift to offset transient length
        pr_sigma_date_transient=2,
        pr_median_lambda=0.08,
        pr_sigma_lambda=0.02,  # No wiggle
    )
)
cp_a.append(  # Lockdown end in two weeks
    dict(
        pr_mean_date_transient=datetime.datetime(2020, 11, 22),
        pr_sigma_date_transient=5,
        pr_median_lambda=1 / 8,
        pr_sigma_lambda=0.02,  # No wiggle
    )
)
log.info(
    f"Szenario 1: Lockdown from {cp_a[-2]['pr_mean_date_transient']} to {cp_a[-1]['pr_mean_date_transient']}"
)


cp_b.append(  # Lockdown start in 2 weeks
    dict(
        pr_mean_date_transient=datetime.datetime(2020, 11, 15)
        + datetime.timedelta(days=1),  # shift to offset transient length
        pr_sigma_date_transient=2,
        pr_median_lambda=0.08,
        pr_sigma_lambda=0.02,  # No wiggle
    )
)
cp_b.append(  # Lockdown start in 2 weeks
    dict(
        pr_mean_date_transient=datetime.datetime(2020, 12, 6),
        pr_sigma_date_transient=5,
        pr_median_lambda=1 / 8,
        pr_sigma_lambda=0.02,  # No wiggle
    )
)
log.info(
    f"Szenario 2: Lockdown from {cp_b[-2]['pr_mean_date_transient']} to {cp_b[-1]['pr_mean_date_transient']}"
)
""" ## Put the model together
"""


def create_model(change_points, params_model):
    with cov19.model.Cov19Model(**params_model) as model:
        # Create the an array of the time dependent infection rate lambda
        lambda_t_log = cov19.model.lambda_t_with_sigmoids(
            pr_median_lambda_0=0.4,
            pr_sigma_lambda_0=0.5,
            change_points_list=change_points,  # The change point priors we constructed earlier
            name_lambda_t="lambda_t",  # Name for the variable in the trace (see later)
        )
        # set prior distribution for the recovery rate
        mu = pm.Lognormal(name="mu", mu=np.log(1 / 8), sigma=0.01)

        # This builds a decorrelated prior for I_begin for faster inference.
        # It is not necessary to use it, one can simply remove it and use the default argument
        # for pr_I_begin in cov19.SIR
        prior_I = cov19.model.uncorrelated_prior_I(
            lambda_t_log=lambda_t_log,
            mu=mu,
            pr_median_delay=10,
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
            pr_mean_of_median=10,
            pr_sigma_of_median=0.2,
            pr_median_of_width=0.5,
        )

        # Modulate the inferred cases by a abs(sin(x)) function, to account for weekend effects
        # Also adds the "new_cases" variable to the trace that has all model features.
        new_cases = cov19.model.week_modulation(
            cases=new_cases, name_cases="new_cases",
        )

        # Define the likelihood, uses the new_cases_obs set as model parameter
        cov19.model.student_t_likelihood(new_cases)

    return model


# Number of days the simulation starts earlier than the data.
# Should be significantly larger than the expected delay in order to always fit the same number of data points.
diff_data_sim = 16
# Number of days in the future (after date_end_data) to forecast cases
num_days_forecast = 80
params_model = dict(
    new_cases_obs=new_cases_obs[:],
    data_begin=data_begin,
    fcast_len=num_days_forecast,
    diff_data_sim=diff_data_sim,
    N_population=83e6,
)


mod_a = create_model(cp_a, params_model)
mod_b = create_model(cp_b, params_model)
mod_c = create_model(cp_c, params_model)

""" ## MCMC sampling
"""
tr_a = pm.sample(model=mod_a, tune=1000, draws=1000, init="advi+adapt_diag")
tr_b = pm.sample(model=mod_b, tune=1000, draws=1000, init="advi+adapt_diag")
tr_c = pm.sample(model=mod_c, tune=1000, draws=1000, init="advi+adapt_diag")

import pickle

pickle.dump(
    [(mod_a, mod_b, mod_c), (tr_a, tr_b, tr_c)],
    open("./data/what_if_lockdown.pickled", "wb"),
)

"""
with open("./data/what_if_lockdown.pickled", "rb") as f:
    [(mod_a, mod_b, mod_c), (tr_a, tr_b, tr_c)] = pickle.load(f)
"""

""" ## Plotting
    
    ### Save path
"""
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
cov19.plot.rcParams.draw_ci_75 = False
cov19.plot.rcParams.draw_ci_95 = False
cov19.plot.rcParams.locale = "de_DE"
cov19.plot.rcParams.date_format = "%d. %b"
cov19.plot.rcParams.fcast_ls = "-"


# Create plots
fig, axes = cov19.plot.timeseries_overview(
    mod_c,
    tr_c,
    offset=total_cases_obs[0],
    forecast_label="Kein Lockdown",
    color="tab:red",
)


fig, axes = cov19.plot.timeseries_overview(
    mod_b,
    tr_b,
    axes=axes,
    offset=total_cases_obs[0],
    forecast_label=f"Lockdown am {datetime.datetime(2020,11,11).strftime(cov19.plot.rcParams.date_format)} (3 Wochen lang)",
    color="tab:orange",
)

fig, axes = cov19.plot.timeseries_overview(
    mod_a,
    tr_a,
    axes=axes,
    offset=total_cases_obs[0],
    forecast_label=f"Lockdown am {datetime.datetime(2020,11,1).strftime(cov19.plot.rcParams.date_format)} (3 Wochen lang)",
    forecast_heading=r"$\bf Szenarien\!:$",
    add_more_later=True,
    color="tab:green",
    start=datetime.datetime(2020, 10, 1),
    end=datetime.datetime(2020, 12, 31),
)

axes[0].set_ylim(-0.07, 0.2)
axes[1].set_ylim(0, 140_000)
axes[2].set_ylim(0, 220_000)

axes[1].set_ylabel("Täglich neue Fallzahlen")
axes[1].set_xlabel("Datum")

# Disable last axes visuals
axes[2].get_xaxis().set_visible(False)
axes[2].get_yaxis().set_visible(False)
axes[2].spines["bottom"].set_visible(False)
axes[2].spines["left"].set_visible(False)
axes[2].texts[0].set_visible(False)  # Remove C letter
legend = axes[2].get_legend()
legend._loc = 10  # center legend
legend.get_texts()[0].set_text("Daten (RKI Meldedatum)")  # Add to Data legend

# Change size of plot
fig.set_size_inches(6, 6)
# Set ratios
fig._gridspecs[0].set_height_ratios([1, 4, 1.5])


fig.savefig(
    save_to + "german_ts.svg",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
    transparent=True,
)
fig.savefig(
    save_to + "german_ts.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
    transparent=True,
)

""" ### Distributions

    Do for each of the 3 models
"""

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
        save_to + "dist_" + name + ".svg",
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
