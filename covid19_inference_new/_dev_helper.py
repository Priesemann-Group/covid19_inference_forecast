# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-04-21 08:57:53
# @Last Modified: 2020-05-20 10:48:23
# ------------------------------------------------------------------------------ #
# Let's have a dummy instance of model and trace so we can play around with the
# interface and plotting.
# Analog to example_bundeslaender.ipynb
# ------------------------------------------------------------------------------ #

import logging

import numpy as np
import pymc3 as pm
import datetime

from .model import *
from . import data_retrieval

log = logging.getLogger(__name__)


def create_example_instance(num_change_points=3):
    """
        Parameters
        ----------
            num_change_points : int
                the standard change points to inlcude, at most 3

        Retruns
        -------
            (model, trace) with example data
    """

    jhu = data_retrieval.JHU()
    jhu.download_all_available_data()

    date_begin_data = datetime.datetime(2020, 3, 10)
    date_end_data = datetime.datetime(2020, 3, 13)

    new_cases_obs = jhu.get_new(
        value="confirmed",
        country="Germany",
        data_begin=date_begin_data,
        data_end=date_end_data,
    )

    diff_data_sim = 16  # should be significantly larger than the expected delay, in
    # order to always fit the same number of data points.
    num_days_forecast = 10

    prior_date_mild_dist_begin = datetime.datetime(2020, 3, 9)
    prior_date_strong_dist_begin = datetime.datetime(2020, 3, 16)
    prior_date_contact_ban_begin = datetime.datetime(2020, 3, 23)

    change_points = [
        dict(
            pr_mean_date_transient=prior_date_mild_dist_begin,
            pr_sigma_date_transient=6,
            pr_median_lambda=0.2,
            pr_sigma_lambda=1,
        ),
        dict(
            pr_mean_date_transient=prior_date_strong_dist_begin,
            pr_sigma_date_transient=6,
            pr_median_lambda=1 / 8,
            pr_sigma_lambda=1,
        ),
        dict(
            pr_mean_date_transient=prior_date_contact_ban_begin,
            pr_sigma_date_transient=6,
            pr_median_lambda=1 / 8 / 2,
            pr_sigma_lambda=1,
        ),
    ]
    change_points = change_points[0:num_change_points]

    params_model = dict(
        new_cases_obs=new_cases_obs,
        data_begin=date_begin_data,
        fcast_len=num_days_forecast,
        diff_data_sim=diff_data_sim,
        N_population=83e6,
    )

    with Cov19Model(**params_model) as model:
        lambda_t_log = lambda_t_with_sigmoids(
            pr_median_lambda_0=0.4, change_points_list=change_points
        )

        new_I_t = SIR(lambda_t_log, mu=0.13)

        new_cases_inferred_raw = delay_cases(new_I_t)

        new_cases_inferred = week_modulation(new_cases_inferred_raw)

        student_t_likelihood(new_cases_inferred)

    # make it fast
    trace = pm.sample(model=model, tune=1, draws=1)

    return model, trace
