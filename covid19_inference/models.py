import datetime

import numpy as np
import pymc3 as pm

from .modelling_help_functions import SIR_model, smooth_step_function, delay_cases

def SIR_with_change_points(change_points_list, date_begin_simulation, num_days_sim, diff_data_sim, new_cases_obs):
    """
    Returns the model with change points
    :param
    change_points_list: list of dics. Each dic has to have the following items:
        'prior_median_λ': float, median λ of the LogNormal prior to which the change point changes to
        'prior_std_λ': float, standard deviation of the LogNormal prior.
        'prior_mean_date_begin_transient': float,
        'prior_std_begin_transient':float
    date_begin_simulation: datetime.datetime. The begin of the simulation data
    :return: model
    """
    with pm.Model() as model:
        # true cases at begin of loaded data but we do not know the real number
        I_begin = pm.HalfCauchy('I_begin', beta=100)

        # fraction of people that are newly infected each day
        λ_list = []
        λ_list.append(pm.Lognormal("λ_0", mu=np.log(0.4), sigma=0.5))
        for i, change_point in enumerate(change_points_list):
            λ_list.append(pm.Lognormal("λ_{}".format(i+1),
                                       mu=np.log(change_point['prior_median_λ']),
                                       sigma=change_point['prior_std_λ']))

        # set the start dates of the two periods
        transient_begin_list = []
        date_before = None
        for i, change_point in enumerate(change_points_list):
            date_begin_transient = change_point['prior_mean_date_begin_transient']
            if date_before is not None and date_before > date_begin_transient:
                raise RuntimeError('Dates of change points are not temporally ordered')
            prior_day_begin_transient = (date_begin_transient - date_begin_simulation).days
            transient_begin = pm.Normal('transient_begin_{}'.format(i), mu=prior_day_begin_transient,
                                        sigma=change_point['prior_std_begin_transient'])
            transient_begin_list.append(transient_begin)
            date_before = date_begin_transient


        # transient time
        transient_len_list=[]
        for i, change_point in enumerate(change_points_list):
            transient_len = pm.Lognormal('transient_len_{}'.format(i),
                                          mu=np.log(change_point['prior_median_λ']),
                                          sigma=change_point['prior_std_λ'])
            transient_len_list.append(transient_len)


        # build the time-dependent spreading rate
        λ_t_list = []
        λ_step_before = λ_list[0]
        for transient_begin, transient_len, λ_step in zip(transient_begin_list,
                                                          transient_len_list,
                                                          λ_list[1:]):
            λ_t = smooth_step_function(λ_begin=0, λ_end=1, t_begin=transient_begin,
                                          t_end=transient_begin + transient_len,
                                          t_total=num_days_sim) * (λ_step - λ_step_before)
            λ_t_list.append(λ_t)
        λ_t = sum(λ_t_list)

        # fraction of people that recover each day, recovery rate mu
        μ = pm.Lognormal('μ', mu=np.log(1 / 8), sigma=0.2)

        # delay in days between contracting the disease and being recorded
        delay = pm.Lognormal("delay", mu=np.log(8), sigma=0.2)

        # prior of the error of observed cases
        σ_obs = pm.HalfCauchy("σ_obs", beta=10)

        N_germany = 83e6

        # -------------------------------------------------------------------------- #
        # training the model with loaded data
        # -------------------------------------------------------------------------- #

        S_begin = N_germany - I_begin
        S, I, new_I = SIR_model(λ=λ_t, μ=μ, S_begin=S_begin, I_begin=I_begin, N=N_germany)

        new_cases_inferred = delay_cases(new_I,
                                         len_new_I_t=num_days_sim,
                                         len_new_cases_obs=num_days_sim - diff_data_sim,
                                         delay=delay, delay_arr=diff_data_sim)
        num_days_data = new_cases_obs.shape[-1]
        # Approximates Poisson
        # calculate the likelihood of the model:
        # observed cases are distributed following studentT around the model
        pm.StudentT(
            "obs",
            nu=4,
            mu=new_cases_inferred[...,:num_days_data],
            sigma=(new_cases_inferred[...,:num_days_data]) ** 0.5 * σ_obs,
            observed=new_cases_obs)

        pm.Deterministic('λ_t', λ_t)
        pm.Deterministic('new_cases', new_cases_inferred)
    return model