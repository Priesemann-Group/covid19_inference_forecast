import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def get_all_free_RVs_names(model):
    varnames = [str(x).replace('_log__', '') for x in model.free_RVs]
    return varnames

def get_prior_distribution(model, x, varname):
    return np.exp(model[varname].distribution.logp(x).eval())

def plot_hist(model, trace, ax, varname, colors = ('tab:blue', 'tab:orange'), bins = 50):
    ax.hist(trace[varname], bins=bins, density=True, color=colors[1],
            label='Posterior')
    limits = ax.get_xlim()
    x = np.linspace(*limits, num=100)
    ax.plot(x, get_prior_distribution(model, x, varname), label='Prior',
            color=colors[0], linewidth=3)
    ax.set_xlim(*limits)
    ax.set_ylabel('Density')
    ax.set_xlabel(varname)


def plot_cases(trace, new_cases_obs, date_begin_sim, diff_data_sim, start_date_plot=None, end_date_plot=None,
               ylim=None):
    """
    Plots the new cases
    :param trace: Needs to have the variables new_cases, delay, μ, λ_t
    :param new_cases_obs:
    :param date_begin_sim:
    :param diff_data_sim:
    :param start_date_plot:
    :param end_date_plot:
    :param ylim:
    :return:
    """
    def conv_time_to_mpl_dates(arr):
        return matplotlib.dates.date2num([datetime.timedelta(days=float(date)) + date_begin_sim for date in arr])

    new_cases_sim = trace.new_cases
    len_sim = trace['λ_t'].shape[1]
    if start_date_plot is None:
        start_date_plot = date_begin_sim + datetime.timedelta(days=diff_data_sim)
    if end_date_plot is None:
        end_date_plot = date_begin_sim + datetime.timedelta(days=len_sim)
    if ylim is None:
        ylim = 1.6*np.max(new_cases_obs)

    num_days_data = len(new_cases_obs)
    diff_to_0 = num_days_data + diff_data_sim
    date_data_end = date_begin_sim + datetime.timedelta(days=diff_data_sim + num_days_data)
    num_days_future = (end_date_plot - date_data_end).days
    start_date_mpl, end_date_mpl = matplotlib.dates.date2num([start_date_plot, end_date_plot])

    color = 'tab:orange'
    fig, axes = plt.subplots(3, 2, figsize=(8, 8), gridspec_kw={'height_ratios': [1, 3, 3],
                                                                'width_ratios': [2, 3]})
    # plt.locator_params(nbins=4)
    pos_letter = (-0.3, 1)
    titlesize = 16

    ax = axes[1][0]
    time_arr = np.arange(-len(new_cases_obs) + 1, 1)
    mpl_dates = conv_time_to_mpl_dates(time_arr) + diff_data_sim + num_days_data
    ax.plot(mpl_dates, new_cases_obs, 'd', markersize=6, label='Data')
    new_cases_past = new_cases_sim[:, :num_days_data]
    percentiles = np.percentile(new_cases_past, q=2.5, axis=0), np.percentile(new_cases_past, q=97.5, axis=0)
    ax.plot(mpl_dates, np.median(new_cases_past, axis=0), color=color, label='Fit (with 95% CI)')
    ax.fill_between(mpl_dates, percentiles[0], percentiles[1], alpha=0.3, color=color)
    ax.set_yscale('log')
    ax.set_ylabel('Number of new cases')
    ax.set_xlabel('Date')
    ax.legend()
    ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(interval=1, byweekday=matplotlib.dates.SU))
    ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d'))
    ax.set_xlim(start_date_mpl)

    ax = axes[1][1]

    time1 = np.arange(-len(new_cases_obs) + 1, 1)
    mpl_dates = conv_time_to_mpl_dates(time1) + diff_data_sim + num_days_data
    ax.plot(mpl_dates, new_cases_obs, 'd', label='Data', markersize=4, color='tab:blue',
            zorder=5)

    new_cases_past = new_cases_sim[:, :num_days_data]
    ax.plot(mpl_dates, np.median(new_cases_past, axis=0), '--', color=color, linewidth=1.5, label='Fit with 95% CI')
    percentiles = np.percentile(new_cases_past, q=2.5, axis=0), np.percentile(new_cases_past, q=97.5, axis=0)
    ax.fill_between(mpl_dates, percentiles[0], percentiles[1], alpha=0.2, color=color)

    time2 = np.arange(0, num_days_future + 1)
    mpl_dates_fut = conv_time_to_mpl_dates(time2) + diff_data_sim + num_days_data
    cases_future = new_cases_sim[:, num_days_data:num_days_data+num_days_future].T
    # cases_future = np.concatenate([np.ones((1,cases_future.shape[1]))*cases_obs[-1], cases_future], axis=0)
    median = np.median(cases_future, axis=-1)
    percentiles = (
        np.percentile(cases_future, q=2.5, axis=-1),
        np.percentile(cases_future, q=97.5, axis=-1),
    )
    ax.plot(mpl_dates_fut[1:], median, color=color, linewidth=3, label='forecast with 75% and 95% CI')
    ax.fill_between(mpl_dates_fut[1:], percentiles[0], percentiles[1], alpha=0.1, color=color)
    ax.fill_between(mpl_dates_fut[1:], np.percentile(cases_future, q=12.5, axis=-1),
                    np.percentile(cases_future, q=87.5, axis=-1),
                    alpha=0.2, color=color)

    ax.set_xlabel('Date')
    ax.set_ylabel('New confirmed cases in Germany')
    ax.legend(loc='upper left')
    ax.set_ylim(0, ylim)
    # ax.legend(loc='lower left')
    # ax.set_xticks([-28,-21, -14, -7, 0, 7, 14, 21, 28])
    # ax.set_xlim(-28, 14)
    # ax.locator_params(axis="y", nbins=4)
    func_format = lambda num, _: "${:.0f}\,$k".format(num / 1_000)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(func_format))
    ax.set_xlim(start_date_mpl, end_date_mpl)
    ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(byweekday=matplotlib.dates.SU))
    ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d'))
    ax.text(pos_letter[0], pos_letter[1], "D", transform=ax.transAxes, size=titlesize)

    ax = axes[0][1]

    time = np.arange(-diff_to_0 + 1, -diff_to_0 + len_sim + 1)
    λ_t = trace['λ_t'][:, :]
    μ = trace['μ'][:, None]
    mpl_dates = conv_time_to_mpl_dates(time) + diff_data_sim + num_days_data

    ax.plot(mpl_dates, np.median(λ_t - μ, axis=0), color=color, linewidth=2)
    ax.fill_between(mpl_dates, np.percentile(λ_t - μ, q=2.5, axis=0), np.percentile(λ_t - μ, q=97.5, axis=0),
                    alpha=0.15,
                    color=color)
    # ax.fill_between(mpl_dates, np.percentile(λ_t , q=12.5, axis=0),np.percentile(λ_t, q=87.5, axis=0), alpha=0.2,
    #                color=color)

    ax.set_ylabel('effective\ngrowth rate $\lambda_t^*$')
    # ax.set_xlabel("days from now")
    # ax.legend(loc='lower left')
    ax.set_xticks([-28, -21, -14, -7, 0, 7, 14, 21, 28])
    # ax.set_xlim(-28, 14)
    ax.set_ylim(-0.15, 0.45)
    # ax.set_yticks([-0.2, 0, 0.2])
    # ax.set_aspect(15, adjustable="box")
    ax.hlines(0, start_date_mpl, end_date_mpl, linestyles=':')
    delay = matplotlib.dates.date2num(date_data_end) - np.percentile(trace.delay, q=75)
    ax.vlines(delay, -10, 10, linestyles='-', colors=['tab:red'])
    # ax.legend()
    ax.text(delay + 0.5, 0.4, 'unconstrained because\nof reporting delay', color='tab:red', verticalalignment='top')
    ax.text(delay - 0.5, 0.4, 'constrained\nby data', color='tab:red', horizontalalignment='right',
            verticalalignment='top')
    ax.text(pos_letter[0], pos_letter[1], "C", transform=ax.transAxes, size=titlesize)
    ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(byweekday=matplotlib.dates.SU))
    ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d'))
    ax.set_xlim(start_date_mpl, end_date_mpl)

    axes[0][0].set_visible(False)

    plt.subplots_adjust(wspace=0.4, hspace=.3)

    return fig, axes