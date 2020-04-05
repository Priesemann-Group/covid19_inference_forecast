import datetime

import numpy as np
import pymc3 as pm
import theano.tensor as tt

from .modelling_help_functions import _SIR_model, _smooth_step_function, _delay_cases, _delay_cases_lognormal, _SEIR_model_with_delay

def SIR_model_with_change_points(new_cases_obs, change_points_list, date_begin_simulation, num_days_sim, diff_data_sim,
                                 priors_dic = None):
    """
    Returns the model with change points
    :param
    change_points_list: list of dics. Each dic has to have the following items:
        'prior_mean_date_begin_transient': float, required

        'prior_median_λ': float, default: 0.4, median λ of the LogNormal prior to which the change point changes to
        'prior_sigma_λ': float, default: 0.5, standard deviation of the LogNormal prior.
        'prior_sigma_begin_transient':float, Default:3
        'prior_median_transient_len': float, Default:3
        'prior_sigma_transient_len': flaot, Defaul:0.3
    date_begin_simulation: datetime.datetime. The begin of the simulation data
    priors_dic: dictionary with the following entries and default values
        prior_beta_I_begin = 100,
        prior_median_λ_0 = 0.4,
        prior_sigma_λ_0 = 0.5,
        prior_median_μ = 1/8,
        prior_sigma_μ = 0.2,
        prior_median_delay = 8,
        prior_sigma_delay = 0.2,
        prior_beta_σ_obs = 10
    :return: model
    """
    if priors_dic is None:
        priors_dic = dict()

    default_priors = dict(prior_beta_I_begin = 100,
                          prior_median_λ_0 = 0.4,
                          prior_sigma_λ_0 = 0.5,
                          prior_median_μ = 1/8,
                          prior_sigma_μ = 0.2,
                          prior_median_delay = 8,
                          prior_sigma_delay = 0.2,
                          prior_beta_σ_obs = 10)
    default_priors_change_points = dict(prior_median_λ = default_priors['prior_median_λ_0'],
                                        prior_sigma_λ = default_priors['prior_sigma_λ_0'],
                                        prior_sigma_date_begin_transient = 3,
                                        prior_median_transient_len = 3,
                                        prior_sigma_transient_len = 0.3,
                                        prior_mean_date_begin_transient = None)

    for prior_name in priors_dic.keys():
        if prior_name not in default_priors:
            raise RuntimeError("Prior with name {} not known".format(prior_name))
    for change_point in change_points_list:
        for prior_name in change_point.keys():
            if prior_name not in default_priors_change_points:
                raise RuntimeError("Prior with name {} not known".format(prior_name))

    for prior_name, value in default_priors.items():
        if prior_name not in priors_dic:
            priors_dic[prior_name] = value
            print('{} was set to default value {}'.format(prior_name, value))
    for prior_name, value in default_priors_change_points.items():
        for i_cp, change_point in enumerate(change_points_list):
            if prior_name not in change_point:
                change_point[prior_name] = value
                print('{} of change point {} was set to default value {}'.format(prior_name, i_cp, value))

    if diff_data_sim < priors_dic['prior_median_delay'] + 3*priors_dic['prior_median_delay']*priors_dic['prior_sigma_delay']:
        raise RuntimeError('diff_data_sim is to small compared to the prior delay')
    if num_days_sim < len(new_cases_obs) + diff_data_sim:
        raise RuntimeError('Simulation ends before the end of the data. Increase num_days_sim.')

    with pm.Model() as model:
        # true cases at begin of loaded data but we do not know the real number
        I_begin = pm.HalfCauchy('I_begin', beta=priors_dic['prior_beta_I_begin'])

        # fraction of people that are newly infected each day
        λ_list = []
        λ_list.append(pm.Lognormal("λ_0", mu=np.log(priors_dic['prior_median_λ_0']), sigma=priors_dic['prior_sigma_λ_0']))
        for i, change_point in enumerate(change_points_list):
            λ_list.append(pm.Lognormal("λ_{}".format(i+1),
                                       mu=np.log(change_point['prior_median_λ']),
                                       sigma=change_point['prior_sigma_λ']))

        # set the start dates of the two periods
        transient_begin_list = []
        date_before = None
        for i, change_point in enumerate(change_points_list):
            date_begin_transient = change_point['prior_mean_date_begin_transient']
            if date_before is not None and date_before > date_begin_transient:
                raise RuntimeError('Dates of change points are not temporally ordered')
            prior_day_begin_transient = (date_begin_transient - date_begin_simulation).days
            transient_begin = pm.Normal('transient_begin_{}'.format(i), mu=prior_day_begin_transient,
                                        sigma=change_point['prior_sigma_date_begin_transient'])
            transient_begin_list.append(transient_begin)
            date_before = date_begin_transient


        # transient time
        transient_len_list=[]
        for i, change_point in enumerate(change_points_list):
            transient_len = pm.Lognormal('transient_len_{}'.format(i),
                                          mu=np.log(change_point['prior_median_transient_len']),
                                          sigma=change_point['prior_sigma_transient_len'])
            transient_len_list.append(transient_len)


        # build the time-dependent spreading rate
        λ_t_list = [λ_list[0]]
        λ_step_before = λ_t_list[0]
        for transient_begin, transient_len, λ_step in zip(transient_begin_list,
                                                          transient_len_list,
                                                          λ_list[1:]):
            λ_t = _smooth_step_function(λ_begin=0, λ_end=1, t_begin=transient_begin,
                                          t_end=transient_begin + transient_len,
                                          t_total=num_days_sim) * (λ_step - λ_step_before)
            λ_step_before = λ_step
            λ_t_list.append(λ_t)
        λ_t = sum(λ_t_list)

        # fraction of people that recover each day, recovery rate mu
        μ = pm.Lognormal('μ', mu=np.log(priors_dic['prior_median_μ']), sigma=priors_dic['prior_sigma_μ'])

        # delay in days between contracting the disease and being recorded
        delay = pm.Lognormal("delay", mu=np.log(priors_dic['prior_median_delay']), sigma=priors_dic['prior_sigma_delay'])

        # prior of the error of observed cases
        σ_obs = pm.HalfCauchy("σ_obs", beta=priors_dic['prior_beta_σ_obs'])

        N_germany = 83e6

        # -------------------------------------------------------------------------- #
        # training the model with loaded data
        # -------------------------------------------------------------------------- #

        S_begin = N_germany - I_begin
        S, I, new_I = _SIR_model(λ=λ_t, μ=μ, S_begin=S_begin, I_begin=I_begin, N=N_germany)

        new_cases_inferred = _delay_cases(new_I,
                                         len_new_I_t=num_days_sim,
                                         len_new_cases_obs=num_days_sim - diff_data_sim,
                                         delay=delay, delay_betw_input_output=diff_data_sim)
        num_days_data = new_cases_obs.shape[-1]
        # Approximates Poisson
        # calculate the likelihood of the model:
        # observed cases are distributed following studentT around the model
        pm.StudentT(
            "obs",
            nu=4,
            mu=new_cases_inferred[:num_days_data],
            sigma=tt.abs_(new_cases_inferred[:num_days_data] + 1) ** 0.5 * σ_obs, # +1 and tt.abs to avoid nans
            observed=new_cases_obs)

        pm.Deterministic('λ_t', λ_t)
        pm.Deterministic('new_cases', new_cases_inferred)
    return model





def more_advanced_model(new_cases_obs, change_points_list, date_begin_simulation, num_days_sim, diff_data_sim,
                        priors_dic = None):
    """
    Returns the model with change points
    :param
    change_points_list: list of dics. Each dic has to have the following items:
        'prior_mean_date_begin_transient': float, required

        'prior_median_λ': float, default: 0.4, median λ of the LogNormal prior to which the change point changes to
        'prior_sigma_λ': float, default: 0.5, standard deviation of the LogNormal prior.
        'prior_sigma_begin_transient':float, Default:3
        'prior_median_transient_len': float, Default:3
        'prior_sigma_transient_len': flaot, Defaul:0.3
    date_begin_simulation: datetime.datetime. The begin of the simulation data
    priors_dic: dictionary with the following entries and default values
        prior_beta_I_begin = 100,
        prior_median_λ_0 = 0.4,
        prior_sigma_λ_0 = 0.5,
        prior_median_μ = 1/8,
        prior_sigma_μ = 0.2,
        prior_median_delay = 8,
        prior_sigma_delay = 0.2,
        prior_beta_σ_obs = 10
    :return: model
    """
    if priors_dic is None:
        priors_dic = dict()

    default_priors = dict(prior_beta_I_begin = 100,
                          prior_beta_E_begin_scale = 10,
                          prior_median_λ_0 = 2,
                          prior_sigma_λ_0 = 0.7,
                          prior_median_μ = 1/3,
                          prior_sigma_μ = 0.3,
                          prior_median_delay = 5,
                          prior_sigma_delay = 0.2,
                          scale_delay = 0.3,
                          prior_beta_σ_obs = 10,
                          prior_σ_random_walk = 0.01,
                          prior_mean_median_incubation = 5, #sources: https://www.ncbi.nlm.nih.gov/pubmed/32150748 , https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7014672/
                                                            #-1 day because persons likely become infectious  before
                          prior_sigma_median_incubation = 1,
                          sigma_incubation = 0.418) # source: https://www.ncbi.nlm.nih.gov/pubmed/32150748

    default_priors_change_points = dict(prior_median_λ = default_priors['prior_median_λ_0'],
                                        prior_sigma_λ = default_priors['prior_sigma_λ_0'],
                                        prior_sigma_date_begin_transient = 3,
                                        prior_median_transient_len = 3,
                                        prior_sigma_transient_len = 0.3,
                                        prior_mean_date_begin_transient = None)

    for prior_name in priors_dic.keys():
        if prior_name not in default_priors:
            raise RuntimeError("Prior with name {} not known".format(prior_name))
    for change_point in change_points_list:
        for prior_name in change_point.keys():
            if prior_name not in default_priors_change_points:
                raise RuntimeError("Prior with name {} not known".format(prior_name))

    for prior_name, value in default_priors.items():
        if prior_name not in priors_dic:
            priors_dic[prior_name] = value
            print('{} was to default value {}'.format(prior_name, value))
    for prior_name, value in default_priors_change_points.items():
        for i_cp, change_point in enumerate(change_points_list):
            if prior_name not in change_point:
                change_point[prior_name] = value
                print('{} of change point {} was set to default value {}'.format(prior_name, i_cp, value))

    if diff_data_sim < priors_dic['prior_median_delay'] + 3*priors_dic['prior_median_delay']*priors_dic['prior_sigma_delay']:
        raise RuntimeError('diff_data_sim is to small compared to the prior delay')
    if num_days_sim < len(new_cases_obs) + diff_data_sim:
        raise RuntimeError('Simulation ends before the end of the data. Increase num_days_sim.')

    with pm.Model() as model:
        # true cases at begin of loaded data but we do not know the real number
        I_begin = pm.HalfCauchy('I_begin', beta=priors_dic['prior_beta_I_begin'])
        E_begin_scale = pm.HalfCauchy('E_begin_scale', beta=priors_dic['prior_beta_E_begin_scale'])
        new_E_begin = pm.HalfCauchy('E_begin', beta=E_begin_scale, shape=9)

        # fraction of people that are newly infected each day
        λ_list = []
        λ_list.append(pm.Lognormal("λ_0", mu=np.log(priors_dic['prior_median_λ_0']), sigma=priors_dic['prior_sigma_λ_0']))
        for i, change_point in enumerate(change_points_list):
            λ_list.append(pm.Lognormal("λ_{}".format(i+1),
                                       mu=np.log(change_point['prior_median_λ']),
                                       sigma=change_point['prior_sigma_λ']))

        # set the start dates of the two periods
        transient_begin_list = []
        date_before = None
        for i, change_point in enumerate(change_points_list):
            date_begin_transient = change_point['prior_mean_date_begin_transient']
            if date_before is not None and date_before > date_begin_transient:
                raise RuntimeError('Dates of change points are not temporally ordered')
            prior_day_begin_transient = (date_begin_transient - date_begin_simulation).days
            transient_begin = pm.Normal('transient_begin_{}'.format(i), mu=prior_day_begin_transient,
                                        sigma=change_point['prior_sigma_date_begin_transient'])
            transient_begin_list.append(transient_begin)
            date_before = date_begin_transient


        # transient time
        transient_len_list=[]
        for i, change_point in enumerate(change_points_list):
            transient_len = pm.Lognormal('transient_len_{}'.format(i),
                                          mu=np.log(change_point['prior_median_transient_len']),
                                          sigma=change_point['prior_sigma_transient_len'])
            transient_len_list.append(transient_len)


        # build the time-dependent spreading rate
        λ_t_list = [λ_list[0] * tt.ones(num_days_sim)]
        λ_step_before = λ_t_list[0]
        for transient_begin, transient_len, λ_step in zip(transient_begin_list,
                                                          transient_len_list,
                                                          λ_list[1:]):
            λ_t = _smooth_step_function(λ_begin=0, λ_end=1, t_begin=transient_begin,
                                          t_end=transient_begin + transient_len,
                                          t_total=num_days_sim) * (λ_step - λ_step_before)
            λ_step_before = λ_step
            λ_t_list.append(λ_t)

        σ_random_walk = pm.HalfNormal('σ_random_walk', sigma=priors_dic['prior_σ_random_walk'])
        λ_t_random_walk = pm.distributions.timeseries.GaussianRandomWalk('λ_t_random_walk', mu=0,
                                                                         sigma=σ_random_walk * tt.ones(num_days_sim),
                                                                         shape = num_days_sim, init = pm.Normal.dist(sigma=0.001))
        λ_t = sum(λ_t_list) + λ_t_random_walk

        # fraction of people that recover each day, recovery rate mu
        μ = pm.Lognormal('μ', mu=np.log(priors_dic['prior_median_μ']), sigma=priors_dic['prior_sigma_μ'])

        # delay in days between contracting the disease and being recorded
        delay = pm.Lognormal("delay", mu=np.log(priors_dic['prior_median_delay']), sigma=priors_dic['prior_sigma_delay'])

        # prior of the error of observed cases
        σ_obs = pm.HalfCauchy("σ_obs", beta=priors_dic['prior_beta_σ_obs'])

        N_germany = 83e6

        # -------------------------------------------------------------------------- #
        # training the model with loaded data
        # -------------------------------------------------------------------------- #

        median_incubation = pm.Normal('median_incubation', mu=priors_dic['prior_mean_median_incubation'],
                                                           sigma=priors_dic['prior_sigma_median_incubation'])
        # sources: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7014672/
        #

        S_begin = N_germany - I_begin
        S_t, new_E_t, I_t, new_I_t = _SEIR_model_with_delay(λ=λ_t, μ=μ,
                                                            S_begin=S_begin, new_E_begin=new_E_begin,
                                                            I_begin=I_begin,
                                                            N=N_germany,
                                                            median_incubation=median_incubation,
                                                            sigma_incubation=0.418) #source: https://www.ncbi.nlm.nih.gov/pubmed/32150748


        new_cases_inferred = _delay_cases_lognormal(input_arr=new_I_t,
                                                    len_input_arr=num_days_sim,
                                                    len_output_arr=num_days_sim - diff_data_sim,
                                                    median_delay=delay,
                                                    scale_delay=priors_dic['scale_delay'],
                                                    delay_betw_input_output=diff_data_sim)
        num_days_data = new_cases_obs.shape[-1]
        # Approximates Poisson
        # calculate the likelihood of the model:
        # observed cases are distributed following studentT around the model
        pm.StudentT(
            "obs",
            nu=4,
            mu=new_cases_inferred[:num_days_data],
            sigma=tt.abs_(new_cases_inferred[:num_days_data] + 1) ** 0.5 * σ_obs, # +1 and tt.abs to avoid nans
            observed=new_cases_obs)

        pm.Deterministic('λ_t', λ_t)
        pm.Deterministic('new_cases', new_cases_inferred)
    return model


