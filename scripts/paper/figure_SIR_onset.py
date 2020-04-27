import datetime
import time as time_module
import sys
import os 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import theano
import matplotlib
import pymc3 as pm
import pickle
import theano.tensor as tt



sys.path.append('../..')
import covid19_inference as cov19

date_data_begin = datetime.date(2020,3,1)
date_data_end = datetime.date(2020,3,15)
num_days_to_predict = 28
num_days_future = 28
diff_data_sim = 16 

date_begin_sim = date_data_begin - datetime.timedelta(days = diff_data_sim)
date_end_sim   = date_data_end   + datetime.timedelta(days = num_days_future)
num_days_sim = (date_end_sim-date_begin_sim).days
num_days_data = (date_data_end-date_data_begin).days

def conv_time_to_mpl_dates(arr):
    try:
        return matplotlib.dates.date2num(
            [datetime.timedelta(days=float(date)) + date_begin_sim for date in arr]
        )
    except:
        return matplotlib.dates.date2num(
            datetime.timedelta(days=float(arr)) + date_begin_sim
        )


def delay_cases(new_I_t, len_new_I_t, len_new_cases_obs , delay, delay_arr):
    """
    Delays the input new_I_t by delay and return and array with length len_new_cases_obs
    The initial delay of the output is set by delay_arr. 
    
    Take care that delay is smaller or equal than delay_arr, otherwise zeros are 
    returned, which could potentially lead to errors

    Also assure that len_new_I_t is larger then len(new_cases_obs)-delay, otherwise it 
    means that the simulated data is not long enough to be fitted to the data.
    """
    delay_mat = make_delay_matrix(n_rows=len_new_I_t, 
                                  n_columns=len_new_cases_obs, initial_delay=delay_arr)
    inferred_cases = interpolate(new_I_t, delay, delay_mat)
    return inferred_cases 

def make_delay_matrix(n_rows, n_columns, initial_delay=0):
    """
    Has in each entry the delay between the input with size n_rows and the output
    with size n_columns
    """
    size = max(n_rows, n_columns)
    mat = np.zeros((size, size))
    for i in range(size):
        diagonal = np.ones(size-i)*(initial_delay + i)
        mat += np.diag(diagonal, i)
    for i in range(1, size):
        diagonal = np.ones(size-i)*(initial_delay - i)
        mat += np.diag(diagonal, -i)
    return mat[:n_rows, :n_columns]

def interpolate(array, delay, delay_matrix):
    interp_matrix = tt.maximum(1-tt.abs_(delay_matrix - delay), 0)
    interpolation = tt.dot(array,interp_matrix)
    return interpolation

def SIR_model(λ, μ, S_begin, I_begin, N):
    new_I_0 = tt.zeros_like(I_begin)
    def next_day(λ, S_t, I_t, _):
        new_I_t = λ/N*I_t*S_t
        S_t = S_t - new_I_t
        I_t = I_t + new_I_t - μ * I_t
        return S_t, I_t, new_I_t
    outputs , _  = theano.scan(fn=next_day, sequences=[λ], 
                               outputs_info=[S_begin, I_begin, new_I_0])
    S_all, I_all, new_I_all = outputs
    return S_all, I_all, new_I_all

def SIR_model_nomu(λ, μ, S_begin, I_begin, N):
    new_I_0 = tt.zeros_like(I_begin)
    def next_day(λ, S_t, I_t, _):
        new_I_t = λ/N*I_t*S_t
        S_t = S_t - new_I_t
        I_t = I_t + new_I_t
        return S_t, I_t, new_I_t
    outputs , _  = theano.scan(fn=next_day, sequences=[λ], 
                               outputs_info=[S_begin, I_begin, new_I_0])
    S_all, I_all, new_I_all = outputs
    return S_all, I_all, new_I_all

def run_model():

    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

    confirmed_cases = pd.read_csv(url, sep=",")

    date_data_begin = datetime.date(2020,3,1)
    date_data_end = datetime.date(2020,3,15)
    num_days_to_predict = 28

    num_days_data = (date_data_end-date_data_begin).days


    diff_data_sim = 16 # should be significantly larger than the expected delay, in 
                       # order to always fit the same number of data points.
    date_begin_sim = date_data_begin - datetime.timedelta(days = diff_data_sim)
    format_date = lambda date_py: '{}/{}/{}'.format(date_py.month, date_py.day,
                                                     str(date_py.year)[2:4])
    date_formatted_begin = format_date(date_data_begin)
    date_formatted_end = format_date(date_data_end)

    cases_obs =  np.array(
        confirmed_cases.loc[confirmed_cases["Country/Region"] == "Germany", 
                            date_formatted_begin:date_formatted_end])[0]
    #cases_obs = np.concatenate([np.nan*np.ones(diff_data_sim), cases_obs])
    print('Cases yesterday ({}): {} and '
          'day before yesterday: {}'.format(date_data_end.isoformat(), *cases_obs[:-3:-1]))
    num_days = (date_data_end - date_begin_sim).days
    date_today = date_data_end + datetime.timedelta(days=1)
    # ------------------------------------------------------------------------------ #
    # model setup and training
    # ------------------------------------------------------------------------------ #
    np.random.seed(0)



    with pm.Model() as model:
        # true cases at begin of loaded data but we do not know the real number
        I_begin = pm.HalfCauchy('I_begin', beta=100)

        # fraction of people that are newly infected each day
        λ = pm.Lognormal("λ", mu=np.log(0.4), sigma=0.5)

        # fraction of people that recover each day, recovery rate mu
        μ = pm.Lognormal('μ', mu=np.log(1/8), sigma=0.2)

        # delay in days between contracting the disease and being recorded
        delay = pm.Lognormal("delay", mu=np.log(8), sigma=0.2)

        # prior of the error of observed cases
        σ_obs = pm.HalfCauchy("σ_obs", beta=10)

        N_germany = 83e6

        # -------------------------------------------------------------------------- #
        # training the model with loaded data
        # -------------------------------------------------------------------------- #

        S_begin = N_germany - I_begin
        S_past, I_past, new_I_past = SIR_model(λ=λ * tt.ones(num_days-1), μ=μ, 
                                                   S_begin=S_begin, I_begin=I_begin,
                                                   N=N_germany)
        new_cases_obs = np.diff(cases_obs)
        new_cases_inferred = delay_cases(new_I_past, len_new_I_t=num_days - 1, 
                                         len_new_cases_obs=len(new_cases_obs), 
                                         delay=delay, delay_arr=diff_data_sim)

        # Approximates Poisson
        # calculate the likelihood of the model:
        # observed cases are distributed following studentT around the model
        pm.StudentT(
            "obs",
            nu=4,
            mu=new_cases_inferred,
            sigma=(new_cases_inferred)**0.5 * σ_obs,
            observed=new_cases_obs)  
        
        S_past = pm.Deterministic('S_past', S_past)
        I_past = pm.Deterministic('I_past', I_past)
        new_I_past = pm.Deterministic('new_I_past', new_I_past)
        new_cases_past = pm.Deterministic('new_cases_past', new_cases_inferred)
        
        # -------------------------------------------------------------------------- #
        # prediction, start with no changes in policy
        # -------------------------------------------------------------------------- #

        S_begin = S_past[-1]
        I_begin = I_past[-1]
        forecast_no_change = SIR_model(λ=λ*tt.ones(num_days_to_predict), μ=μ, 
                            S_begin=S_begin, I_begin=I_begin, N=N_germany)
        S_no_change, I_no_change, new_I_no_change = forecast_no_change

        #saves the variables for later retrieval
        pm.Deterministic('S_no_change', S_no_change)
        pm.Deterministic('I_no_change', I_no_change)
        pm.Deterministic('new_I_no_change', new_I_no_change)

        new_cases_inferred = delay_cases(tt.concatenate([new_I_past[-diff_data_sim:], new_I_no_change]), 
                                         len_new_I_t=diff_data_sim + num_days_to_predict, 
                                         len_new_cases_obs=num_days_to_predict, 
                                         delay=delay, delay_arr=diff_data_sim)
        pm.Deterministic('new_cases_no_change', new_cases_inferred)


        # -------------------------------------------------------------------------- #
        # social distancing, m reduced by about 50 percent
        # -------------------------------------------------------------------------- #
        #For all following predictions:
        length_transient = 7  # days


        # λ is decreased by 50%
        reduc_factor_mild = 0.5
        days_offset = 0  # start the decrease in spreading rate after this

        time_arr = np.arange(num_days_to_predict)

        # change in m along time
        λ_correction = tt.clip((time_arr - days_offset) / length_transient, 0, 1)
        λ_t_soc_dist= λ * (1 - λ_correction * reduc_factor_mild) 

        S_begin = S_past[-1]
        I_begin = I_past[-1]
        forecast_soc_dist = SIR_model(λ=λ_t_soc_dist, μ=μ, 
                            S_begin=S_begin, I_begin=I_begin, 
                            N=N_germany)
        S_soc_dist, I_soc_dist, new_I_soc_dist = forecast_soc_dist
        pm.Deterministic('S_soc_dist', S_soc_dist)
        pm.Deterministic('I_soc_dist', I_soc_dist)
        pm.Deterministic('new_I_soc_dist', new_I_soc_dist)

        new_cases_inferred = delay_cases(tt.concatenate([new_I_past[-diff_data_sim:], new_I_soc_dist]), 
                                        len_new_I_t=diff_data_sim + num_days_to_predict, 
                                        len_new_cases_obs=num_days_to_predict, 
                                        delay=delay, delay_arr=diff_data_sim)
        pm.Deterministic('new_cases_soc_dist', new_cases_inferred)

        # -------------------------------------------------------------------------- #
        # isolation, almost no new infections besides baseline after transient phase
        # -------------------------------------------------------------------------- #

        # λ is decreased by 90%
        reduc_factor_strong = 0.9
        days_offset = 0  # start the decrease in spreading rate after this

        # spreading of people who transmit although they are isolated
        time_arr = np.arange(num_days_to_predict)

        # change in λ along time
        λ_correction = tt.clip((time_arr - days_offset) / length_transient, 0, 1)
        λ_t_isol= λ * (1 - λ_correction * reduc_factor_strong)

        S_begin = S_past[-1]
        I_begin = I_past[-1]
        forecast_isol = SIR_model(λ=λ_t_isol , μ=μ, 
                                  S_begin=S_begin, I_begin=I_begin, 
                                  N=N_germany)
        S_isol, I_isol, new_I_isol = forecast_isol

        pm.Deterministic('S_isol', S_isol)
        pm.Deterministic('I_isol', I_isol)  
        pm.Deterministic('new_I_isol', new_I_isol)

        new_cases_inferred = delay_cases(tt.concatenate([new_I_past[-diff_data_sim:], new_I_isol]), 
                                    len_new_I_t=diff_data_sim + num_days_to_predict, 
                                    len_new_cases_obs=num_days_to_predict, 
                                    delay=delay, delay_arr=diff_data_sim)
        pm.Deterministic('new_cases_isol', new_cases_inferred)

        # -------------------------------------------------------------------------- #
        # isolation 5 days later, almost no new infections besides baseline after transient phase
        # -------------------------------------------------------------------------- #

        # λ is decreased by 90%
        reduc_factor_strong = 0.9
        days_offset = 5  # start the decrease in spreading rate after this

        # spreading of people who transmit although they are isolated
        time_arr = np.arange(num_days_to_predict)

        # change in λ along time
        λ_correction = tt.clip((time_arr - days_offset) / length_transient, 0, 1)
        λ_t_isol_later= λ * (1 - λ_correction * reduc_factor_strong) 

        S_begin = S_past[-1]
        I_S_beginbegin = I_past[-1]
        forecast_isol_later = SIR_model(λ=λ_t_isol_later, μ=μ, 
                             S_begin=S_begin, I_begin=I_begin, 
                             N=N_germany)
        S_isol_later, I_isol_later, new_I_isol_later = forecast_isol_later

        pm.Deterministic('S_isol_later', S_isol_later)
        pm.Deterministic('I_isol_later', I_isol_later)  
        pm.Deterministic('new_I_isol_later', new_I_isol_later)

        new_cases_inferred = delay_cases(tt.concatenate([new_I_past[-diff_data_sim:], new_I_isol_later]), 
                                len_new_I_t=diff_data_sim + num_days_to_predict, 
                                len_new_cases_obs=num_days_to_predict, 
                                delay=delay, delay_arr=diff_data_sim)
        pm.Deterministic('new_cases_isol_later', new_cases_inferred)


        # -------------------------------------------------------------------------- #
        # isolation 5 days earlier, almost no new infections besides baseline after transient phase
        # -------------------------------------------------------------------------- #

        length_transient = 7

        # λ is decreased by 90%
        reduc_factor = 0.9
        days_offset = -5  # start the decrease in spreading rate after this

        # spreading of people who transmit although they are isolated
        time_arr = np.arange(days_offset, num_days_to_predict)

        # change in λ along time

        λ_t_earlier  = tt.clip((time_arr-days_offset) / length_transient, 0, 1)*\
                          (λ*(1-reduc_factor) - λ) + λ


        S_begin = S_past[-1 + days_offset]
        I_begin = I_past[-1 + days_offset]
        forecast_earlier = SIR_model(λ=λ_t_earlier, μ=μ, 
                             S_begin=S_begin, I_begin=I_begin, 
                             N=N_germany)
        S_earlier, I_earlier, new_I_earlier = forecast_earlier

        pm.Deterministic('S_earlier', S_earlier)
        pm.Deterministic('I_earlier', I_earlier)  
        pm.Deterministic('new_I_earlier', new_I_earlier)
        pm.Deterministic('λ_t_earlier', λ_t_earlier)


        new_cases_inferred = delay_cases(tt.concatenate([new_I_past[-diff_data_sim:days_offset], new_I_earlier]), 
                                len_new_I_t=diff_data_sim + num_days_to_predict, 
                                len_new_cases_obs=num_days_to_predict, 
                                delay=delay, delay_arr=diff_data_sim)
        
        pm.Deterministic('new_cases_earlier', new_cases_inferred)


        # -------------------------------------------------------------------------- #
        # long transient scenario
        # -------------------------------------------------------------------------- #

        length_transient = 14

        # λ is decreased by 90%
        reduc_factor = 0.9
        days_offset = -3.5  # start the decrease in spreading rate after this
        days_offset_sim = -4

        # spreading of people who transmit although they are isolated
        time_arr = np.arange(days_offset_sim, num_days_to_predict)

        # change in λ along time

        λ_t_long_trans  = tt.clip((time_arr-days_offset) / length_transient, 0, 1)*\
                          (λ*(1-reduc_factor) - λ) + λ


        S_begin = S_past[-1 + days_offset_sim]
        I_begin = I_past[-1 + days_offset_sim]
        forecast_long_trans = SIR_model(λ=λ_t_long_trans, μ=μ, 
                             S_begin=S_begin, I_begin=I_begin, 
                             N=N_germany)
        S_long_trans, I_long_trans, new_I_long_trans = forecast_long_trans

        pm.Deterministic('S_long_trans', S_long_trans)
        pm.Deterministic('I_long_trans', I_long_trans)  
        pm.Deterministic('new_I_long_trans', new_I_long_trans)
        pm.Deterministic('λ_t_long_trans', λ_t_long_trans)


        new_cases_inferred = delay_cases(tt.concatenate([new_I_past[-diff_data_sim:days_offset_sim], new_I_long_trans]), 
                                len_new_I_t=diff_data_sim + num_days_to_predict, 
                                len_new_cases_obs=num_days_to_predict, 
                                delay=delay, delay_arr=diff_data_sim)
        pm.Deterministic('new_cases_long_trans', new_cases_inferred)


        # -------------------------------------------------------------------------- #
        # immediate transient scenario
        # -------------------------------------------------------------------------- #

        # λ is decreased by 90%
        reduc_factor_strong = 0.9
        days_offset = 3.5 # start the decrease in spreading rate after this
        length_transient = 0.5

        # spreading of people who transmit although they are isolated
        time_arr = np.arange(num_days_to_predict)

        # change in λ along time
        λ_correction = tt.clip((time_arr - days_offset) / length_transient, 0, 1)
        λ_t_isol= λ * (1 - λ_correction * reduc_factor_strong)

        S_begin = S_past[-1]
        I_begin = I_past[-1]
        forecast_isol = SIR_model(λ=λ_t_isol , μ=μ, 
                                  S_begin=S_begin, I_begin=I_begin, 
                                  N=N_germany)
        S_isol, I_isol, new_I_isol = forecast_isol

        pm.Deterministic('S_immedi', S_isol)
        pm.Deterministic('I_immedi', I_isol)  
        pm.Deterministic('new_immedi', new_I_isol)

        new_cases_inferred = delay_cases(tt.concatenate([new_I_past[-diff_data_sim:], new_I_isol]), 
                                    len_new_I_t=diff_data_sim + num_days_to_predict, 
                                    len_new_cases_obs=num_days_to_predict, 
                                    delay=delay, delay_arr=diff_data_sim)
        pm.Deterministic('new_cases_immedi', new_cases_inferred)

        # -------------------------------------------------------------------------- #
        # run model, pm trains and predicts when calling this
        # -------------------------------------------------------------------------- #
        
        trace = pm.sample(draws=3000, tune=800, chains=2)

    return trace, cases_obs

def create_figure_timeseries(
    trace,
    cases_obs,
    color="tab:green",
    save_to=None,
    num_days_futu_to_plot=18,
    y_lim_lambda=(-0.15, 0.45),
    plot_red_axis=True,
    axes=None,
    forecast_label="Forecast",
    add_more_later=False,
    country='Germany',
    num_days_data=15,
    diff_data_sim = 16,
):
    """
        Used for the generation of the timeseries forecast figure around easter on the
        repo.

        Parameters
        ----------
        trace: trace instance
            needed for the data
        color: string
            main color to use, default "tab:green"
        save_to: string or None
            path where to save the figures. default: None, not saving figures
        num_days_futu_to_plot : int
            how many days to plot into the future (not exceeding simulation)
        y_lim_lambda : (float, float)
            min, max values for lambda effective. default (-0.15, 0.45)
        plot_red_axis : bool
            show the unconstrained constrained annotation in lambda panel
        axes : np.array of mpl axes
            provide an array of existing axes (from previously calling this function)
            to add more traces. Data will not be added again. Ideally call this first
            with `add_more_later=True`
        forecast_label : string
            legend label for the forecast, default: "Forecast"
        add_more_later : bool
            set this to true if you plan to add multiple models to the plot. changes the layout (and the color of the fit to past data)

        Returns
        -------
            fig : mpl figure
            axes : np array of mpl axeses (insets not included)

    """

    plot_par = dict()
    plot_par["draw_insets_cases"] = True
    plot_par["draw_ci_95"] = True
    plot_par["draw_ci_75"] = False
    plot_par["insets_only_two_ticks"] = True

    axes_provided = False
    if axes is not None:
        print("Provided axes, adding new content")
        axes_provided = True

    ylabel_new = f"Daily new reported\ncases in {country}"
    ylabel_cum = f"Total reported\ncases in {country}"
    ylabel_lam = f"Effective\ngrowth rate $\lambda^\\ast (t)$"

    pos_letter = (-0.3, 1)
    titlesize = 16
    insetsize = ("25%", "50%")
    figsize = (4, 6)
    # figsize = (6, 6)

    leg_loc = "upper left"
    if plot_par["draw_insets_cases"] == True:
        leg_loc = "upper right"

    new_c_ylim = [0, 15_000]
    new_c_insetylim = [50, 17_000]

    cum_c_ylim = [0, 300_000]
    cum_c_insetylim = [50, 300_000]

    color_futu = color
    color_past = color
    if axes_provided:
        fig = axes[0].get_figure()
    else:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=figsize,
            gridspec_kw={"height_ratios": [3, 3]},
            constrained_layout=True,
        )
        if add_more_later:
            color_past = "#646464"

    insets = []

    diff_to_0 = num_days_data + diff_data_sim

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

    # --------------------------------------------------------------------------- #
    # prepare data
    # --------------------------------------------------------------------------- #
    # observed data, only one dim: [day]
    new_c_obsd = np.diff(cases_obs)
    cum_c_obsd = cases_obs

    # model traces, dims: [sample, day],
    new_c_past = trace["new_cases_no_change"][:, :num_days_data]
    new_c_futu = trace["new_cases_no_change"][
        :, num_days_data : num_days_data + num_days_futu_to_plot
    ]
    cum_c_past = np.cumsum(np.insert(new_c_past, 0, 0, axis=1), axis=1) + cases_obs[0]
    cum_c_futu = np.cumsum(np.insert(new_c_futu, 0, 0, axis=1), axis=1) + cases_obs[-1]

    ax = axes[0]
    if not axes_provided:
        ax.text(
            pos_letter[0], pos_letter[1], "B", transform=ax.transAxes, size=titlesize
        )
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
            color=color_past,
            linewidth=1.5,
            label="Fit",
            zorder=10,
        )
        if plot_par["draw_ci_95"] == True:
            ax.fill_between(
                mpl_dates_past[1:],
                np.percentile(new_c_past, q=2.5, axis=0),
                np.percentile(new_c_past, q=97.5, axis=0),
                alpha=0.1,
                color=color_past,
                lw=0,
            )
        # dummy element to separate forecasts
        if add_more_later:
            ax.plot(
                [],
                [],
                "-",
                linewidth=0,
                label="Forecasts:",
                # fontweight="bold"
            )

    ax.plot(
        mpl_dates_futu[1:],
        np.median(new_c_futu, axis=0),
        "--",
        color=color_futu,
        linewidth=3,
        label=forecast_label,
    )
    if plot_par["draw_ci_95"] == True:
        ax.fill_between(
            mpl_dates_futu[1:],
            np.percentile(new_c_futu, q=2.5, axis=0),
            np.percentile(new_c_futu, q=97.5, axis=0),
            alpha=0.1,
            color=color_futu,
            lw=0,
        )
    if plot_par["draw_ci_75"] == True:
        ax.fill_between(
            mpl_dates_futu[1:],
            np.percentile(new_c_futu, q=12.5, axis=0),
            np.percentile(new_c_futu, q=87.5, axis=0),
            alpha=0.2,
            color=color_futu,
            lw=0,
        )
    ax.set_ylabel(ylabel_new)
    ax.set_ylim(new_c_ylim)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_k))

    # NEW CASES LOG SCALE, skip forecast
    if plot_par["draw_insets_cases"] == True:
        ax = inset_axes(
            ax, width=insetsize[0], height=insetsize[1], loc=2, borderpad=1
        )
        insets.append(ax)
        if not axes_provided:
            ax.plot(
                mpl_dates_past[1:],
                new_c_obsd,
                "d",
                markersize=2,
                label="Data",
                zorder=5,
            )
        ax.plot(
            mpl_dates_past[1:],
            np.median(new_c_past, axis=0),
            "-",
            color=color_past,
            label="Fit",
            zorder=10,
        )
        if plot_par["draw_ci_95"] == True:
            ax.fill_between(
                mpl_dates_past[1:],
                np.percentile(new_c_past, q=2.5, axis=0),
                np.percentile(new_c_past, q=97.5, axis=0),
                alpha=0.1,
                color=color_past,
                lw=0,
            )
        # ax.set_yticks([1e1, 1e2, 1e3, 1e4, 1e5])
        ax.set_ylim(new_c_insetylim)

    # --------------------------------------------------------------------------- #
    # Total cases, lin scale first
    # --------------------------------------------------------------------------- #
    ax = axes[1]
    if not axes_provided:
        ax.text(
            pos_letter[0], pos_letter[1], "C", transform=ax.transAxes, size=titlesize
        )
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
            color=color_past,
            linewidth=1.5,
            label="Fit",
            zorder=10,
        )
        if plot_par["draw_ci_95"] == True:
            ax.fill_between(
                mpl_dates_past[:],
                np.percentile(cum_c_past, q=2.5, axis=0),
                np.percentile(cum_c_past, q=97.5, axis=0),
                alpha=0.1,
                color=color_past,
                lw=0,
            )
        # dummy element to separate forecasts
        if add_more_later:
            ax.plot(
                [],
                [],
                "-",
                linewidth=0,
                label="Forecasts:",
                # fontweight="bold"
            )

    ax.plot(
        mpl_dates_futu[1:],
        np.median(cum_c_futu[:, 1:], axis=0),
        "--",
        color=color_futu,
        linewidth=3,
        label=f"{forecast_label}",
    )
    if plot_par["draw_ci_95"] == True:
        ax.fill_between(
            mpl_dates_futu[1:],
            np.percentile(cum_c_futu[:, 1:], q=2.5, axis=0),
            np.percentile(cum_c_futu[:, 1:], q=97.5, axis=0),
            alpha=0.1,
            color=color_futu,
            lw=0,
        )
    if plot_par["draw_ci_75"] == True:
        ax.fill_between(
            mpl_dates_futu[1:],
            np.percentile(cum_c_futu[:, 1:], q=12.5, axis=0),
            np.percentile(cum_c_futu[:, 1:], q=87.5, axis=0),
            alpha=0.2,
            color=color_futu,
            lw=0,
        )
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel_cum)
    ax.set_ylim(cum_c_ylim)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_k))

    # Total CASES LOG SCALE, skip forecast
    if plot_par["draw_insets_cases"] == True:
        ax = inset_axes(
            ax, width=insetsize[0], height=insetsize[1], loc=2, borderpad=1
        )
        insets.append(ax)
        if not axes_provided:
            ax.plot(
                mpl_dates_past[:], cum_c_obsd, "d", markersize=2, label="Data", zorder=5
            )
        ax.plot(
            mpl_dates_past[:],
            np.median(cum_c_past, axis=0),
            "-",
            color=color_past,
            label="Fit",
            zorder=10,
        )
        if plot_par["draw_ci_95"] == True:
            ax.fill_between(
                mpl_dates_past[:],
                np.percentile(cum_c_past, q=2.5, axis=0),
                np.percentile(cum_c_past, q=97.5, axis=0),
                alpha=0.1,
                color=color_past,
                lw=0,
            )
        # ax.set_yticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
        ax.set_ylim(cum_c_insetylim)

    # --------------------------------------------------------------------------- #
    # Finalize
    # --------------------------------------------------------------------------- #

    for ax in axes:
        ax.set_rasterization_zorder(rasterization_zorder)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlim(start_date, end_date)
        format_date_xticks(ax)

        # biweekly, remove every second element
        if not axes_provided:
            for label in ax.xaxis.get_ticklabels()[1::2]:
                label.set_visible(False)

    for ax in insets:
        ax.set_xlim(start_date, mid_date)
        ax.yaxis.tick_right()
        ax.set_yscale("log")
        if plot_par["insets_only_two_ticks"] is True:
            format_date_xticks(ax, minor=False)
            ax.set_xticks([ax.get_xticks()[0], ax.get_xticks()[-1]])
            for label in ax.xaxis.get_ticklabels()[1:-1]:
                label.set_visible(False)
        else:
            format_date_xticks(ax)
            for label in ax.xaxis.get_ticklabels()[1:-1]:
                label.set_visible(False)
    insets[0].set_yticks(
        [1e2, 1e3, 1e4,]
    )
    insets[1].set_yticks(
        [1e2, 1e4, 1e6,]
    )

    # crammed data, disable some more tick labels
    insets[0].xaxis.get_ticklabels()[-1].set_visible(False)
    insets[0].yaxis.get_ticklabels()[0].set_visible(False)

    # legend
    ax = axes[1]
    ax.legend(loc=leg_loc)
    ax.get_legend().get_frame().set_linewidth(0.0)
    ax.get_legend().get_frame().set_facecolor("#F0F0F0")

    # add_watermark(axes[1])

    # fig.suptitle(
    #     f"Latest forecast\n({datetime.datetime.now().strftime('%Y/%m/%d')})",
    #     x=0.15,
    #     y=1.075,
    #     verticalalignment="top",
    #     fontweight="bold",
    # )

    axes[1].set_title(
        f"Data until {date_data_end.strftime('%B %-d')}",
        loc="right",
        fontweight="bold",
        fontsize="small"
    )

    # plt.subplots_adjust(wspace=0.4, hspace=0.25)
    if save_to is not None:
        plt.savefig(
            save_to + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0,
        )
        plt.savefig(
            save_to + ".png", dpi=300, bbox_inches="tight", pad_inches=0,
        )