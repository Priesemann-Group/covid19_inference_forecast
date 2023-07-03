import datetime
import time as time_module
import sys
import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import theano
import matplotlib
import pymc3 as pm
import theano.tensor as tt


try:
    import covid19_inference as cov19
except ModuleNotFoundError:
    sys.path.append('../..')
    import covid19_inference as cov19

path_to_save = 'figures/'
path_save_pickled = 'data/'
rerun = True

# Download data and uses local fallback if unsuccessful: "../data/confirmed_global_fallback_2020-04-28.csv"
confirmed_cases = cov19.get_jhu_confirmed_cases()

country = 'Germany'
date_data_begin = datetime.datetime(2020,3,1)
date_data_end = datetime.datetime(2020,4,21)

num_days_data = (date_data_end-date_data_begin).days
diff_data_sim = 16 # should be significantly larger than the expected delay, in
                   # order to always fit the same number of data points.
num_days_future = 28
date_begin_sim = date_data_begin - datetime.timedelta(days = diff_data_sim)
date_end_sim   = date_data_end   + datetime.timedelta(days = num_days_future)
num_days_sim = (date_end_sim-date_begin_sim).days


cases_obs = cov19.filter_one_country(confirmed_cases, country,
                                     date_data_begin, date_data_end)

print('Cases yesterday ({}): {} and '
      'day before yesterday: {}'.format(date_data_end.isoformat(), *cases_obs[:-3:-1]))

prior_date_mild_dist_begin =  datetime.datetime(2020,3,9)
prior_date_strong_dist_begin =  datetime.datetime(2020,3,16)
prior_date_contact_ban_begin =  datetime.datetime(2020,3,23)

change_points = [dict(pr_mean_date_begin_transient = prior_date_mild_dist_begin,
                      pr_sigma_date_begin_transient = 3,
                      pr_median_lambda = 0.2,
                      pr_sigma_lambda = 0.5),
                 dict(pr_mean_date_begin_transient = prior_date_strong_dist_begin,
                      pr_sigma_date_begin_transient = 1,
                      pr_median_lambda = 1/8,
                      pr_sigma_lambda = 0.5),
                 dict(pr_mean_date_begin_transient = prior_date_contact_ban_begin,
                      pr_sigma_date_begin_transient = 1,
                      pr_median_lambda = 1/8/2,
                      pr_sigma_lambda = 0.5)]
if rerun:

    traces = []
    models = []
    for num_change_points in [3]:
        model = cov19.SIR_with_change_points(new_cases_obs = np.diff(cases_obs),
                                            change_points_list = change_points[:num_change_points],
                                            date_begin_simulation = date_begin_sim,
                                            num_days_sim = num_days_sim,
                                            diff_data_sim = diff_data_sim,
                                            N = 83e6,
                                            priors_dict=None,
                                            weekends_modulated=True,
                                            weekend_modulation_type = 'abs_sine')
        models.append(model)
        traces.append(pm.sample(model=model, init='advi', draws=400, tune=100))


    pickle.dump([models, traces], open(path_save_pickled + 't1_100.pickled', 'wb'))



    traces = []
    models = []
    for num_change_points in [3]:
        model = cov19.SIR_with_change_points(new_cases_obs = np.diff(cases_obs),
                                            change_points_list = change_points[:num_change_points],
                                            date_begin_simulation = date_begin_sim,
                                            num_days_sim = num_days_sim,
                                            diff_data_sim = diff_data_sim,
                                            N = 83e6,
                                            priors_dict=None,
                                            weekends_modulated=True,
                                            weekend_modulation_type = 'abs_sine',
                                            time_step=0.1)
        models.append(model)
        traces.append(pm.sample(model=model, init='advi', draws=400, tune=100))


    pickle.dump([models, traces], open(path_save_pickled + 't01_100.pickled', 'wb'))
