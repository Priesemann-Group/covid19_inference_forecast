 
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


# Set Latex and fonts
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = \
    r'\usepackage{{amsmath}}\renewcommand{\sfdefault}{phv}'


try:
    import covid19_inference as cov19
except ModuleNotFoundError:
    sys.path.append('../..')
    import covid19_inference as cov19

path_to_save = 'figures/'
path_save_pickled = 'data/'
rerun = True

def conv_time_to_mpl_dates(arr, date_begin_sim):
    try:
        return matplotlib.dates.date2num(
            [datetime.timedelta(days=float(date)) + date_begin_sim for date in arr]
        )
    except:
        return matplotlib.dates.date2num(
            datetime.timedelta(days=float(arr)) + date_begin_sim
        )
def format_date_xticks(ax, minor=None):
    ax.xaxis.set_major_locator(
        matplotlib.dates.WeekdayLocator(interval=1, byweekday=matplotlib.dates.SU)
    )
    if minor is None:
        minor = True
    if minor:
        ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %-d"))


def plot():
    with open(path_save_pickled + 't1_100.pickled', 'rb') as f:
        models, traces = pickle.load(f)
    trace = traces[0]

    r1 = pm.diagnostics.gelman_rubin(traces[0])
    worst_r = 0.0
    for k, v in r1.items():
        try:
            worst_r = max(worst_r, max(v))
        except TypeError:
            worst_r = max(worst_r, v)
    print('rhat,', worst_r)

    with open(path_save_pickled + 't01_100.pickled', 'rb') as f:
        models, traces = pickle.load(f)
    trace_acc = traces[0]

    r1 = pm.diagnostics.gelman_rubin(traces[0])
    worst_r = 0.0
    for k, v in r1.items():
        try:
            worst_r = max(worst_r, max(v))
        except TypeError:
            worst_r = max(worst_r, v)
    print('rhat,', worst_r)

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

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 2)

    num_days_futu_to_plot=10
    diff_to_0 = num_days_data + diff_data_sim

    # interval for the plots with forecast
    start_date = conv_time_to_mpl_dates(-len(cases_obs) + 2, date_begin_sim) + diff_to_0
    end_date = conv_time_to_mpl_dates(num_days_futu_to_plot, date_begin_sim) + diff_to_0
    mid_date = conv_time_to_mpl_dates(1, date_begin_sim) + diff_to_0

    # x-axis for dates, new_cases are one element shorter than cum_cases, use [1:]
    # 0 is the last recorded data point
    time_past = np.arange(-len(cases_obs) + 1, 1)
    time_futu = np.arange(0, num_days_futu_to_plot + 1)
    mpl_dates_past = conv_time_to_mpl_dates(time_past, date_begin_sim) + diff_to_0
    mpl_dates_futu = conv_time_to_mpl_dates(time_futu, date_begin_sim) + diff_to_0

    # --------------------------------------------------------------------------- #
    # prepare data
    # --------------------------------------------------------------------------- #
    # observed data, only one dim: [day]
    new_c_obsd = np.diff(cases_obs)
    cum_c_obsd = cases_obs

    # model traces, dims: [sample, day],
    new_c_past = trace["new_cases"][:, :num_days_data]
    new_c_futu = trace["new_cases"][
        :, num_days_data-1 : num_days_data + num_days_futu_to_plot
    ]
    cum_c_past = np.cumsum(np.insert(new_c_past, 0, 0, axis=1), axis=1) + cases_obs[0]
    cum_c_futu = np.cumsum(np.insert(new_c_futu, 0, 0, axis=1), axis=1) + cases_obs[-1]

    # --------------------------------------------------------------------------- #
    # growth rate lambda*
    # --------------------------------------------------------------------------- #

    color_futu = 'k'

    mu = trace["mu"][:, np.newaxis]
    lambda_t = trace["lambda_t"][
        :, diff_data_sim : diff_data_sim + num_days_data + num_days_futu_to_plot
    ]
    r_end = 14
    ax.plot(
        np.concatenate([mpl_dates_past[1:], mpl_dates_futu[1:]])[:-r_end],
        np.median(lambda_t / mu, axis=0)[:-r_end],
        color=color_futu,
        linewidth=2,
        label=r'$\Delta t=1.0$'
    )
    ax.fill_between(
        np.concatenate([mpl_dates_past[1:], mpl_dates_futu[1:]])[:-r_end],
        np.percentile(lambda_t / mu, q=2.5, axis=0)[:-r_end],
        np.percentile(lambda_t / mu, q=97.5, axis=0)[:-r_end],
        alpha=0.15,
        color=color_futu,
        lw=0
    )

    mu = trace_acc["mu"][:, np.newaxis]
    lambda_t = trace_acc["lambda_t"][
        :, diff_data_sim : diff_data_sim + num_days_data + num_days_futu_to_plot
    ]

    ax.plot(
        np.concatenate([mpl_dates_past[1:], mpl_dates_futu[1:]])[:-r_end],
        np.median(lambda_t / mu, axis=0)[:-r_end],
        color='royalblue',
        linewidth=2,
        ls='-.',
        label=r'$\Delta t=0.1$'
    )
    ax.fill_between(
        np.concatenate([mpl_dates_past[1:], mpl_dates_futu[1:]])[:-r_end],
        np.percentile(lambda_t / mu, q=2.5, axis=0)[:-r_end],
        np.percentile(lambda_t / mu, q=97.5, axis=0)[:-r_end],
        alpha=0.15,
        color='royalblue',
        lw=0
    )
    ax.set_xlim(start_date, end_date - r_end)
    format_date_xticks(ax)
    # biweekly, remove every second element
    for label in ax.xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    ax.legend()
    ax.set_ylabel(r'$R_0$')



    ax = fig.add_subplot(1, 2, 1)
    ax.plot(
        mpl_dates_past[1:],
        new_c_obsd,
        "o",
        label="Data",
        markersize=2,
        color="k",
        zorder=5,
    )
    ax.plot(
        mpl_dates_past[1:],
        np.median(new_c_past, axis=0),
        "-",
        color='k',
        linewidth=1.5,
        label=r"Fit, $\Delta t=1.0$",
        zorder=10,
    )
    ax.fill_between(
        mpl_dates_past[1:],
        np.percentile(new_c_past, q=2.5, axis=0),
        np.percentile(new_c_past, q=97.5, axis=0),
        alpha=0.1,
        color='k',
        lw=0,
    )
    ax.plot(
        mpl_dates_futu[0:],
        np.median(new_c_futu, axis=0),
        "-",
        color='k',
        linewidth=1.5
    )
    ax.fill_between(
        mpl_dates_futu[0:],
        np.percentile(new_c_futu, q=2.5, axis=0),
        np.percentile(new_c_futu, q=97.5, axis=0),
        alpha=0.1,
        color='k',
        lw=0,
    )



    # model traces, dims: [sample, day],
    new_c_past = trace_acc["new_cases"][:, :num_days_data]
    new_c_futu = trace_acc["new_cases"][
        :, num_days_data-1 : num_days_data + num_days_futu_to_plot
    ]
    ax.plot(
        mpl_dates_past[1:],
        np.median(new_c_past, axis=0),
        "-.",
        color='royalblue',
        linewidth=1.5,
        label=r"Fit, $\Delta t=0.1$",
        zorder=-10,
    )
    ax.fill_between(
        mpl_dates_past[1:],
        np.percentile(new_c_past, q=2.5, axis=0),
        np.percentile(new_c_past, q=97.5, axis=0),
        alpha=0.1,
        color='royalblue',
        lw=0,
        zorder=-10
    )
    ax.plot(
        mpl_dates_futu[0:],
        np.median(new_c_futu, axis=0),
        "-.",
        color='royalblue',
        linewidth=1.5,
        zorder=-1
    )
    ax.fill_between(
        mpl_dates_futu[0:],
        np.percentile(new_c_futu, q=2.5, axis=0),
        np.percentile(new_c_futu, q=97.5, axis=0),
        alpha=0.1,
        color='royalblue',
        lw=0,
        zorder=-10
    )



    ax.set_xlim(start_date, end_date)
    format_date_xticks(ax)
    # biweekly, remove every second element
    for label in ax.xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    ax.set_ylabel('Cases')

    ax.legend()
    plt.show()


if __name__ == '__main__':
    plot()
