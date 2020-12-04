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
data_begin = datetime.datetime(2020, 8, 1)
data_end = datetime.datetime(2020, 11, 1)
rki = cov19.data_retrieval.RKI(True)
# rki.download_all_available_data(force_download=True)
new_cases_obs = rki.get_new("confirmed", data_begin=data_begin, data_end=data_end)
total_cases_obs = rki.get_total("confirmed", data_begin=data_begin, data_end=data_end)


""" ## Create weekly changepoints up to the 12. Oktober
"""
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
log.info(f"Adding possible change points at:")
for i, day in enumerate(pd.date_range(start=data_begin, end=data_end)):
    if day.weekday() == 0 and day < datetime.datetime(2020, 11, 1):
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

cp_a.append(  # Lockdown streng
    dict(
        pr_mean_date_transient=datetime.datetime(2020, 11, 2)
        + datetime.timedelta(days=1),  # shift to offset transient length
        pr_sigma_date_transient=2,
        pr_median_lambda=0.04,
        pr_sigma_lambda=0.02,  # No wiggle
    )
)

cp_b.append(  # Lockdown mild 2.nov
    dict(
        pr_mean_date_transient=datetime.datetime(2020, 11, 2)
        + datetime.timedelta(days=1),  # shift to offset transient length
        pr_sigma_date_transient=2,
        pr_median_lambda=1 / 8,
        pr_sigma_lambda=0.02,  # No wiggle
    )
)
# cp_c no lockdown


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
tr_a = pm.sample(model=mod_a, tune=100, draws=100, init="advi+adapt_diag")
tr_b = pm.sample(model=mod_b, tune=100, draws=100, init="advi+adapt_diag")
tr_c = pm.sample(model=mod_c, tune=100, draws=100, init="advi+adapt_diag")

import pickle

pickle.dump(
    [(mod_a, mod_b, mod_c), (tr_a, tr_b, tr_c)],
    open("./data/what_if_lockdown_nov.pickled", "wb"),
)
