import datetime

import numpy as np
import pymc3 as pm
import theano.tensor as tt

# theano.config.gcc.cxxflags = "-Wno-c++11-narrowing" # workaround for macos

import model_helper as mh


def SIR_model_with_change_points(
    new_cases_obs,
    change_points_list,
    date_begin_simulation,
    num_days_sim,
    diff_data_sim,
    N,
    priors_dict=None,
):
    """
        Parameters
        ----------
        new_cases_obs : list or array
            Timeseries (day over day) of newly reported cases (not the total number)

        change_points_list: list of dicts.
            Each dict has to have the following items (with default values):
                pr_mean_date_begin_transient :  datetime.datetime, required
                pr_beta_I_begin :               number, default = 100
                pr_median_lambda_0 :            number, default = 0.4
                pr_sigma_lambda_0 :             number, default = 0.5
                pr_median_mu :                  number, default = 1 / 8
                pr_sigma_mu :                   number, default = 0.2
                pr_median_delay :               number, default = 8
                pr_sigma_delay :                number, default = 0.2
                pr_beta_sigma_obs :             number, default = 10

        date_begin_simulation: datetime.datetime. The begin of the simulation data

        num_days_sim : integer
            Number of days to forecast into the future

        diff_data_sim : integer
            Number of days that the simulation-begin predates the first data point in
            `new_cases_obs`. This is necessary so the model can fit the reporting delay.
            Set this parameter to a value larger than what you expect to find
            for the reporting delay.

        N : number
            The population size. For Germany, we used 83e6

        priors_dict: dictionary with the following entries
            pr_beta_I_begin :        number, default = 100
            pr_median_lambda_0 :     number, default = 0.4
            pr_sigma_lambda_0 :      number, default = 0.5
            pr_median_mu :           number, default = 1/8
            pr_sigma_mu :            number, default = 0.2
            pr_median_delay :        number, default = 8
            pr_sigma_delay :         number, default = 0.2
            pr_beta_sigma_obs :      number, default = 10

        Returns
        -------
        : pm.Model
            Returns an instance of pymc3 model with the change points

        Example
        -------
    """
    if priors_dict is None:
        priors_dict = dict()

    default_priors = dict(
        pr_beta_I_begin=100,
        pr_median_lambda_0=0.4,
        pr_sigma_lambda_0=0.5,
        pr_median_mu=1 / 8,
        pr_sigma_mu=0.2,
        pr_median_delay=8,
        pr_sigma_delay=0.2,
        pr_beta_sigma_obs=10,
    )
    default_priors_change_points = dict(
        pr_median_lambda=default_priors["pr_median_lambda_0"],
        pr_sigma_lambda=default_priors["pr_sigma_lambda_0"],
        pr_sigma_date_begin_transient=3,
        pr_median_transient_len=3,
        pr_sigma_transient_len=0.3,
        pr_mean_date_begin_transient=None,
    )

    for prior_name in priors_dict.keys():
        if prior_name not in default_priors:
            raise RuntimeError(f"Prior with name {prior_name} not known")
    for change_point in change_points_list:
        for prior_name in change_point.keys():
            if prior_name not in default_priors_change_points:
                raise RuntimeError(f"Prior with name {prior_name} not known")

    for prior_name, value in default_priors.items():
        if prior_name not in priors_dict:
            priors_dict[prior_name] = value
            print(f"{prior_name} was set to default value {value}")
    for prior_name, value in default_priors_change_points.items():
        for i_cp, change_point in enumerate(change_points_list):
            if prior_name not in change_point:
                change_point[prior_name] = value
                print(
                    f"{prior_name} of change point {i_cp} was set to default value {value}"
                )

    if (
        diff_data_sim
        < priors_dict["pr_median_delay"]
        + 3 * priors_dict["pr_median_delay"] * priors_dict["pr_sigma_delay"]
    ):
        raise RuntimeError("diff_data_sim is to small compared to the prior delay")
    if num_days_sim < len(new_cases_obs) + diff_data_sim:
        raise RuntimeError(
            "Simulation ends before the end of the data. Increase num_days_sim."
        )

    # ------------------------------------------------------------------------------ #
    # Model and prior implementation
    # ------------------------------------------------------------------------------ #

    with pm.Model() as model:
        # all pm functions now apply on the model instance
        # true cases at begin of loaded data but we do not know the real number
        I_begin = pm.HalfCauchy("I_begin", beta=priors_dict["pr_beta_I_begin"])

        # fraction of people that are newly infected each day
        lambda_list = []
        lambda_list.append(
            pm.Lognormal(
                name="lambda_0",
                mu=np.log(priors_dict["pr_median_lambda_0"]),
                sigma=priors_dict["pr_sigma_lambda_0"],
            )
        )
        for i, cp in enumerate(change_points_list):
            lambda_list.append(
                pm.Lognormal(
                    name=f"lambda_{i + 1}",
                    mu=np.log(cp["pr_median_lambda"]),
                    sigma=cp["pr_sigma_lambda"],
                )
            )

        # list of start dates of the transient periods of the change points
        tr_begin_list = []
        dt_before = date_begin_simulation
        for i, cp in enumerate(change_points_list):
            dt_begin_transient = cp["pr_mean_date_begin_transient"]
            if dt_before is not None and dt_before > dt_begin_tr:
                raise RuntimeError("Dates of change points are not temporally ordered")

            prior_mean = (
                dt_begin_transient - date_begin_simulation
            ).days  # convert the provided date format (argument) into days (a number)

            tr_begin = pm.Normal(
                name=f"transient_begin_{i}",
                mu=prior_mean,
                sigma=cp["pr_sigma_date_begin_transient"],
            )
            tr_begin_list.append(tr_begin)
            dt_before = dt_begin_transient

        # same for transient times
        tr_len_list = []
        for i, cp in enumerate(change_points_list):
            tr_len = pm.Lognormal(
                name=f"transient_len_{i}",
                mu=np.log(cp["pr_median_transient_len"]),
                sigma=cp["pr_sigma_transient_len"],
            )
            tr_len_list.append(tr_len)

        # build the time-dependent spreading rate
        lambda_t_list = [lambda_list[0]]
        lambda_before = lambda_t_list[0]

        for tr_begin, tr_len, lambda_after in zip(
            tr_begin_list, tr_len_list, lambda_list[1:]
        ):
            lambda_t = mh.smooth_step_function(
                start_val=0,
                end_val=1,
                t_begin=tr_begin,
                t_end=tr_begin + tr_len,
                t_total=num_days_sim,
            ) * (lambda_after - lambda_before)
            lambda_before = lambda_after
            lambda_t_list.append(lambda_t)
        lambda_t = sum(lambda_t_list)

        # fraction of people that recover each day, recovery rate mu
        mu = pm.Lognormal(
            name="mu",
            mu=np.log(priors_dict["pr_median_mu"]),
            sigma=priors_dict["pr_sigma_mu"],
        )

        # delay in days between contracting the disease and being recorded
        delay = pm.Lognormal(
            name="delay",
            mu=np.log(priors_dict["pr_median_delay"]),
            sigma=priors_dict["pr_sigma_delay"],
        )

        # prior of the error of observed cases
        sigma_obs = pm.HalfCauchy("sigma_obs", beta=priors_dict["pr_beta_sigma_obs"])

        # -------------------------------------------------------------------------- #
        # training the model with loaded data provided as argument
        # -------------------------------------------------------------------------- #

        S_begin = N - I_begin
        S, I, new_I = _SIR_model(
            lambda_t=lambda_t, mu=mu, S_begin=S_begin, I_begin=I_begin, N=N
        )

        new_cases_inferred = mh.delay_cases(
            new_I_t=new_I,
            len_new_I_t=num_days_sim,
            len_out=num_days_sim - diff_data_sim,
            delay=delay,
            delay_diff=diff_data_sim,
        )
        num_days_data = new_cases_obs.shape[-1]

        # likelihood of the model:
        # observed cases are distributed following studentT around the model.
        # we want to approximate a Poisson distribution of new cases.
        # we choose nu=4 to get heavy tails and robustness to outliers.
        # https://www.jstor.org/stable/2290063
        pm.StudentT(
            name="_new_cases_studentT",
            nu=4,
            mu=new_cases_inferred[:num_days_data],
            sigma=tt.abs_(new_cases_inferred[:num_days_data] + 1) ** 0.5
            * sigma_obs,  # +1 and tt.abs to avoid nans
            observed=new_cases_obs,
        )

        # add these observables to the model so we can extract a time series of them
        # later via e.g. `model.trace['lambda_t']`
        pm.Deterministic("lambda_t", lambda_t)
        pm.Deterministic("new_cases", new_cases_inferred)

    return model


def _SIR_model(lambda_t, mu, S_begin, I_begin, N):
    new_I_0 = tt.zeros_like(I_begin)

    def next_day(lambda_t, S_t, I_t, _, mu, N):
        new_I_t = lambda_t / N * I_t * S_t
        S_t = S_t - new_I_t
        I_t = I_t + new_I_t - mu * I_t
        I_t = tt.clip(I_t, 0, N)  # for stability
        return S_t, I_t, new_I_t

    outputs, _ = theano.scan(
        fn=next_day,
        sequences=[lambda_t],
        outputs_info=[S_begin, I_begin, new_I_0],
        non_sequences=[mu, N],
    )
    S_all, I_all, new_I_all = outputs
    return S_all, I_all, new_I_all


def _SEIR_model_with_delay(
    lambda_t,
    mu,
    S_begin,
    new_E_begin,
    I_begin,
    N,
    median_incubation=5,
    sigma_incubation=0.418,
):
    """

    """
    x = np.arange(1, 9)
    beta = mh.tt_lognormal(x, tt.log(median_incubation), sigma_incubation)
    new_I_0 = tt.zeros_like(I_begin)

    def next_day(
        lambda_t, S_t, nE1, nE2, nE3, nE4, nE5, nE6, nE7, nE8, I_t, _, mu, beta, N
    ):
        new_E_t = lambda_t / N * I_t * S_t
        S_t = S_t - new_E_t
        new_I_t = (
            beta[0] * nE1
            + beta[1] * nE2
            + beta[2] * nE3
            + beta[3] * nE4
            + beta[4] * nE5
            + beta[5] * nE6
            + beta[6] * nE7
            + beta[7] * nE8
        )
        I_t = I_t + new_I_t - mu * I_t
        I_t = tt.clip(I_t, 0.001, N)  # for stability
        # I_t = tt.nnet.sigmoid(I_t/N)*N
        # S_t = tt.clip(S_t, -1, N+1)
        return S_t, new_E_t, I_t, new_I_t

    outputs, _ = theano.scan(
        fn=next_day,
        sequences=[lambda_t],
        outputs_info=[
            S_begin,
            dict(initial=new_E_begin, taps=[-1, -2, -3, -4, -5, -6, -7, -8]),
            I_begin,
            new_I_0,
        ],
        non_sequences=[mu, beta, N],
    )
    S_all, new_E_all, I_all, new_I_all = outputs
    return S_all, new_E_all, I_all, new_I_all
