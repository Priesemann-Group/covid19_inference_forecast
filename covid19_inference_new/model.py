import datetime
import platform
import logging

import theano
import theano.tensor as tt
import numpy as np
import pymc3 as pm
from pymc3 import Model  # this import is needed to get pymc3-style "with ... as model:"

from . import model_helper as mh

log = logging.getLogger(__name__)

if platform.system() == 'Darwin':
    theano.config.gcc.cxxflags = "-Wno-c++11-narrowing" # workaround for macos


class Cov19Model(Model):
    """
        Model class used to create a covid-19 propagation dynamics model

        Parameters
        ----------
        new_cases_obs : 1 or 2d array
            If the array is two-dimensional, an hierarchical model will be constructed. First dimension is then time,
            the second the region/country.
        date_begin_data : datatime.datetime
            Date of the first data point
        num_days_forecast : int
            Number of days the simulations runs longer than the data
        diff_data_sim : int
            Number of days the simulation starts earlier than the data. Should be significantly longer than the delay
            between infection and report of cases.
        N_population : number or 1d array
            Number of inhabitance in region, needed for the S(E)IR model. Is ideally 1 dimensional if new_cases_obs is 2 dimensional
        name : string
            suffix appended to the name of random variables saved in the trace
        model :
            specify a model, if this one should expand another

        Example
        -------
        .. code-block::

            with Cov19Model(**params) as model:
                # Define model here


    """

    def __init__(
        self,
        new_cases_obs,
        date_begin_data,
        num_days_forecast,
        diff_data_sim,
        N_population,
        name="",
        model=None,
    ):

        super().__init__(name=name, model=model)

        # first dim time, second might be state
        self.new_cases_obs = np.array(new_cases_obs)
        self.sim_ndim = new_cases_obs.ndim
        self.N_population = N_population

        # these are dates specifying the bounds of data, simulation and forecast.
        # Jonas Sebastian and Paul agreed to use fully inclusive intervals this makes
        # calculating ranges a bit harder but function arguments are more intuitive.
        # 01 Mar, 02 Mar, 03 Mar
        # data_begin = 01 Mar
        # data_end = 03 Mar
        # [data_begin, data_end]
        # (data_end - data_begin).days = 2

        self.data_begin = date_begin_data
        self.sim_begin = self.data_begin - datetime.timedelta(days=diff_data_sim)
        self.data_end = self.data_begin + datetime.timedelta(
            days=len(new_cases_obs) - 1
        )
        self.sim_end = self.data_end + datetime.timedelta(days=num_days_forecast)

        # totel length of simulation, get later via the shape
        sim_len = len(new_cases_obs) + diff_data_sim + num_days_forecast
        if sim_len < len(new_cases_obs) + diff_data_sim:
            raise RuntimeError(
                "Simulation ends before the end of the data. Increase num_days_sim."
            )

        # shape and dimension of simulation
        if self.sim_ndim == 1:
            self.sim_shape = (sim_len,)
        elif self.sim_ndim == 2:
            self.sim_shape = (sim_len, self.new_cases_obs.shape[1])

    # helper properties
    @property
    def sim_diff_data(self):
        return (self.data_begin - self.sim_begin).days

    @property
    def fcast_begin(self):
        return self.data_end + datetime.timedelta(days=1)

    @property
    def fcast_end(self):
        return self.sim_end

    @property
    def fcast_len(self):
        return (self.sim_end - self.data_end).days

    @property
    def data_len(self):
        return self.new_cases_obs.shape[0]

    @property
    def sim_len(self):
        return self.sim_shape[0]


def modelcontext(model):
    """
        return the given model or try to find it in the context if there was
        none supplied.
    """
    if model is None:
        return Cov19Model.get_context()
    return model


def student_t_likelihood(
    new_cases_inferred, pr_beta_sigma_obs=30, nu=4, offset_sigma=1, model=None,
    data_obs = None, name_student_t = '_new_cases_studentT'
):
    """
        Set the likelihood to apply to the model observations (`model.new_cases_obs`)
        We assume a student-t distribution, the mean of the distribution matches `new_cases_inferred` as provided.

        Parameters
        ----------
        new_cases_inferred : array
            One or two dimensonal array.
            If 2 dimensional, the first dimension is time and the second are the
            regions/countries

        pr_beta_sigma_obs : float

        nu : float
            How flat the tail of the distribution is. Larger nu should  make the model
            more robust to outliers

        offset_sigma : float

        model:
            The model on which we want to add the distribution

        data_obs : array
            The data that is observed. By default it is ``model.new_cases_ob``

        name_student_t :
            The name under which the studentT distribution is saved in the trace.

        Returns
        -------
        None

        TODO
        ----
        #@jonas, can we make it more clear that this whole stuff gets attached to the
        # model? like the with model as context...
        #@jonas doc description for sigma parameters

    """

    model = modelcontext(model)

    len_sigma_obs = () if model.sim_ndim == 1 else model.sim_shape[1]
    sigma_obs = pm.HalfCauchy("sigma_obs", beta=pr_beta_sigma_obs, shape=len_sigma_obs)

    if data_obs is None:
        data_obs = model.new_cases_obs

    pm.StudentT(
        name=name_student_t,
        nu=nu,
        mu=new_cases_inferred[: len(data_obs)],
        sigma=tt.abs_(new_cases_inferred[: len(data_obs)] + offset_sigma) ** 0.5
        * sigma_obs,  # offset and tt.abs to avoid nans
        observed=data_obs,
    )


def SIR(
    lambda_t_log,
    pr_beta_I_begin=100,
    pr_median_mu=1 / 8,
    pr_sigma_mu=0.2,
    model=None,
    return_all=False,
    save_all=False,
):
    r"""
        Implements the susceptible-infected-recovered model.

        .. math::

            I_{new}(t) &= \lambda_t I(t-1)  \frac{S(t-1)}{N}   \\
            S(t) &= S(t-1) - I_{new}(t)  \\
            I(t) &= I(t-1) + I_{new}(t) - \mu  I(t)

        The prior distribution of the recovery rate :math:`\mu` is set to
        :math:`LogNormal(\text{log(pr\_median\_mu)), pr\_sigma\_mu})`. And the prior distribution of
        :math:`I(0)` to :math:`HalfCauchy(\text{pr\_beta\_I\_begin})`

        Parameters
        ----------
        lambda_t_log : :class:`~theano.tensor.TensorVariable`
            time series of the logarithm of the spreading rate, 1 or 2-dimensional. If 2-dimensional the first
            dimension is time.

        pr_beta_I_begin : float or array_like
            Prior beta of the Half-Cauchy distribution of :math:`I(0)`.

        pr_median_mu : float or array_like
            Prior for the median of the lognormal distrubution of the recovery rate :math:`\mu`.

        pr_sigma_mu : float or array_like
            Prior for the sigma of the lognormal distribution of recovery rate :math:`\mu`.

        model : :class:`Cov19Model`
            if none, it is retrieved from the context

        return_all : bool
            if True, returns ``new_I_t``, ``I_t``, ``S_t`` otherwise returns only ``new_I_t``
        save_all : bool
            if True, saves ``new_I_t``, ``I_t``, ``S_t`` in the trace, otherwise it saves only ``new_I_t``

        Returns
        -------

        new_I_t : array
            time series of the number daily newly infected persons.
        I_t : array
            time series of the infected (if return_all set to True)
        S_t : array
            time series of the susceptible (if return_all set to True)

    """
    model = modelcontext(model)

    # Build prior distributions:
    mu = pm.Lognormal(name="mu", mu=np.log(pr_median_mu), sigma=pr_sigma_mu)

    # Total number of people in population
    N = model.N_population

    # Number of regions as tuple of int
    num_regions = () if model.sim_ndim == 1 else model.sim_shape[1]

    # Prior distributions of starting populations (infectious, susceptibles)
    I_begin = pm.HalfCauchy(name="I_begin", beta=pr_beta_I_begin, shape=num_regions)
    S_begin = N - I_begin

    lambda_t = tt.exp(lambda_t_log)
    new_I_0 = tt.zeros_like(I_begin)

    # Runs SIR model:
    def next_day(lambda_t, S_t, I_t, _, mu, N):
        new_I_t = lambda_t / N * I_t * S_t
        S_t = S_t - new_I_t
        I_t = I_t + new_I_t - mu * I_t
        I_t = tt.clip(I_t, 0, N)  # for stability
        S_t = tt.clip(S_t, 0, N)
        return S_t, I_t, new_I_t

    # theano scan returns two tuples, first one containing a time series of
    # what we give in outputs_info : S, I, new_I
    outputs, _ = theano.scan(
        fn=next_day,
        sequences=[lambda_t],
        outputs_info=[S_begin, I_begin, new_I_0],
        non_sequences=[mu, N],
    )
    S_t, I_t, new_I_t = outputs
    pm.Deterministic("new_I_t", new_I_t)
    if save_all:
        pm.Deterministic("S_t", S_t)
        pm.Deterministic("I_t", I_t)

    if return_all:
        return new_I_t, I_t, S_t
    else:
        return new_I_t


def SEIR(
    lambda_t_log,
    pr_beta_I_begin=100,
    pr_beta_new_E_begin=50,
    pr_median_mu=1 / 8,
    median_incubation=5,
    sigma_incubation=0.418,
    pr_sigma_mu=0.2,
    model=None,
    return_all=False,
    save_all=False,
):
    """
        Implements the susceptible-exposed-infected-recovered model

        Parameters
        ----------
        lambda_t_log : 1 or 2d theano tensor
            time series of the logarithm of the spreading rate
        pr_beta_I_begin : int
            scale parameter value for prior distribution of starting number of infectious population
        pr_beta_new_E_begin : int
            scale parameter value for prior distribution of starting number of newly exposed population
            assumed smaller than total infectious population
        pr_median_mu : float
            median recovery rate; parameter value for prior distribution of recovery rate
        pr_sigma_mu : float
            scale parameter value for prior distribution of recovery rate

        model : :class:`Cov19Model`
            if none, it is retrieved from the context

        return_all : Bool
            if True, returns new_I_t, I_t, S_t otherwise returns only new_I_t
        save_all : Bool
            if True, saves new_I_t, I_t, S_t in the trace, otherwise it saves only new_I_t

        Returns
        -------

        new_I_t : array
            time series of the new infected
        I_t : array
            time series of the infected (if return_all set to True)
        S_t : array
            time series of the susceptible (if return_all set to True)
    """
    model = modelcontext(model)

    # Build prior distrubutions:
    # --------------------------

    # Prior distribution of recovery rate mu
    mu = pm.Lognormal(name="mu", mu=np.log(pr_median_mu), sigma=pr_sigma_mu,)

    # Total number of people in population
    N = model.N_population

    # Number of regions as tuple of int
    num_regions = () if model.sim_ndim == 1 else model.sim_shape[1]

    # Prior distributions of starting populations (exposed, infectious, susceptibles)
    # We choose to consider the transitions of newly exposed people of the last 8 days.
    if num_regions == ():
        new_E_begin = pm.HalfCauchy(name="E_begin", beta=pr_beta_new_E_begin, shape=11)
    else:
        new_E_begin = pm.HalfCauchy(
            name="E_begin", beta=pr_beta_new_E_begin, shape=(11, num_regions)
        )
    I_begin = pm.HalfCauchy(name="I_begin", beta=pr_beta_I_begin, shape=num_regions)
    S_begin = N - I_begin - pm.math.sum(new_E_begin, axis=0)

    lambda_t = tt.exp(lambda_t_log)
    new_I_0 = tt.zeros_like(I_begin)

    # Choose transition rates (E to I) according to incubation period distribution
    if num_regions == ():
        x = np.arange(1, 11)
    else:
        x = np.arange(1, 11)[:, None]
        median_incubation = median_incubation*tt.ones(num_regions)
        sigma_incubation = sigma_incubation*tt.ones(sigma_incubation)

    beta = mh.tt_lognormal(x, tt.log(median_incubation), sigma_incubation)

    # Runs SEIR model:
    def next_day(
        lambda_t, S_t, nE1, nE2, nE3, nE4, nE5, nE6, nE7, nE8, nE9, nE10, I_t, _, mu, beta, N
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
            + beta[8] * nE9
            + beta[9] * nE10
        )
        I_t = I_t + new_I_t - mu * I_t
        I_t = tt.clip(I_t, 0, N)  # for stability
        S_t = tt.clip(S_t, 0, N)
        return S_t, new_E_t, I_t, new_I_t

    # theano scan returns two tuples, first one containing a time series of
    # what we give in outputs_info : S, E's, I, new_I
    outputs, _ = theano.scan(
        fn=next_day,
        sequences=[lambda_t],
        outputs_info=[
            S_begin,
            dict(initial=new_E_begin, taps=[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]),
            I_begin,
            new_I_0,
        ],
        non_sequences=[mu, beta, N],
    )
    S_t, nE, I_t, new_I_t = outputs
    pm.Deterministic("new_I_t", new_I_t)
    if save_all:
        pm.Deterministic("S_t", S_t)
        pm.Deterministic("I_t", I_t)

    if return_all:
        return new_I_t, I_t, S_t
    else:
        return new_I_t


def delay_cases(
    new_I_t,
    pr_median_delay=10,
    pr_sigma_delay=0.2,
    pr_median_scale_delay=0.3,
    pr_sigma_scale_delay=None,
    model=None,
    save_in_trace=True,
    name_delay="delay",
    name_delayed_cases="new_cases_raw",
    len_output_arr = None,
    diff_input_output = None
):
    r"""
        Convolves the input by a lognormal distribution, in order to model a delay:

        .. math::

            y_\text{delayed}(t) &= \sum_{\tau=0}^T y_\text{input}(\tau) LogNormal[log(\text{delay}), \text{pr\_median\_scale\_delay}](t - \tau)\\
            log(\text{delay}) &= Normal(log(\text{pr\_sigma\_delay}), \text{pr\_sigma\_delay})

        For clarification: the :math:`LogNormal` distribution is a function evaluated at :math:`t - \tau`.

        If the model is 2-dimensional, the :math:`log(\text{delay})` is hierarchically modelled with the
        :func:`hierarchical_normal` function using the default parameters except that the
        prior :math:`\sigma` of :math:`\text{delay}_\text{L2}` is HalfNormal distributed (``error_cauchy=False``).


        Parameters
        ----------
        new_I_t : :class:`~theano.tensor.TensorVariable`
            The input, typically the number newly infected cases :math:`I_{new}(t)` of from the output of
            :func:`SIR` or :func:`SEIR`.
        pr_median_delay : float
            The prior of the median delay
        scale_delay : float
            The scale of the delay, that is how wide the distribution is.
        pr_sigma_delay : float
            The prior for the sigma of the median delay distribution.
        model : :class:`Cov19Model`
            if none, it is retrieved from the context
        save_in_trace : bool
            whether to save :math:`y_\text{delayed}` in the trace
        name_delay : str
            The name under which the delay is saved in the trace, suffixes and prefixes are added depending on which
            variable is saved.
        name_delayed_cases : str
            The name under which the delay is saved in the trace, suffixes and prefixes are added depending on which
            variable is saved.
        len_output_arr : int
            Length of the array returned. By default it set to the length of the cases_obs saved in the model plus
            the number of days of the forecast.
        diff_input_output : int
            Number of days the returned array begins later then the input. Should be significantly larger than
            the median delay. By default it is set to the ``model.sim_diff_data``.

        Returns
        -------
        new_cases_inferred : :class:`~theano.tensor.TensorVariable`
            The delayed input :math:`y_\text{delayed}(t)`, typically the daily number new cases that one expects to measure.
    """

    model = modelcontext(model)

    if len_output_arr is None:
        len_output_arr = model.data_len + model.fcast_len
    if diff_input_output is None:
        diff_input_output = model.sim_diff_data

    len_delay = () if model.sim_ndim == 1 else model.sim_shape[1]
    delay_L2_log, delay_L1_log = hierarchical_normal(
        name_delay + "_log",
        "sigma_" + name_delay,
        np.log(pr_median_delay),
        pr_sigma_delay,
        len_delay,
        w=0.9,
        error_cauchy=False,
    )
    if delay_L1_log is not None:
        pm.Deterministic(f"{name_delay}_L2", np.exp(delay_L2_log))
        pm.Deterministic(f"{name_delay}_L1", np.exp(delay_L1_log))
    else:
        pm.Deterministic(f"{name_delay}", np.exp(delay_L2_log))

    if pr_sigma_scale_delay is not None:
        scale_delay_L2_log, scale_delay_L1_log = hierarchical_normal(
            "scale_" + name_delay,
            "sigma_scale_" + name_delay,
            np.log(pr_median_scale_delay),
            pr_sigma_scale_delay,
            len_delay,
            w=0.9,
            error_cauchy=False,
        )
        if scale_delay_L1_log is not None:
            pm.Deterministic(f"scale_{name_delay}_L2", tt.exp(scale_delay_L2_log))
            pm.Deterministic(f"scale_{name_delay}_L1", tt.exp(scale_delay_L1_log))

        else:
            pm.Deterministic(f"scale_{name_delay}", tt.exp(scale_delay_L2_log))
    else:
        scale_delay_L2_log = np.log(pr_median_scale_delay)

    new_cases_inferred = mh.delay_cases_lognormal(
        input_arr=new_I_t,
        len_input_arr=model.sim_len,
        len_output_arr=len_output_arr,
        median_delay=tt.exp(delay_L2_log),
        scale_delay=tt.exp(scale_delay_L2_log),
        delay_betw_input_output=diff_input_output,
    )
    if save_in_trace:
        pm.Deterministic(name_delayed_cases, new_cases_inferred)

    return new_cases_inferred


def week_modulation(
    new_cases_inferred,
    week_modulation_type="abs_sine",
    pr_mean_weekend_factor=0.7,
    pr_sigma_weekend_factor=0.2,
    week_end_days=(6, 7),
    model=None,
    save_in_trace=True,
):
    """

    Parameters
    ----------
    new_cases_inferred
    week_modulation_type
    pr_mean_weekend_factor
    pr_sigma_weekend_factor
    week_end_days
    model

    Returns
    -------

    """
    model = modelcontext(model)
    shape_modulation = list(model.sim_shape)
    shape_modulation[0] -= model.sim_diff_data

    len_L2 = () if model.sim_ndim == 1 else model.sim_shape[1]

    week_end_factor, _ = hierarchical_normal(
        "weekend_factor",
        "sigma_weekend_factor",
        pr_mean=pr_mean_weekend_factor,
        pr_sigma=pr_sigma_weekend_factor,
        len_L2=len_L2,
    )
    if week_modulation_type == "step":
        modulation = np.zeros(shape_modulation[0])
        for i in range(shape_modulation[0]):
            date_curr = model.data_begin + datetime.timedelta(days=i)
            if date_curr.isoweekday() in week_end_days:
                modulation[i] = 1
    elif week_modulation_type == "abs_sine":
        offset_rad = pm.VonMises("offset_modulation_rad", mu=0, kappa=0.01)
        offset = pm.Deterministic("offset_modulation", offset_rad / (2 * np.pi) * 7)
        t = np.arange(shape_modulation[0]) - model.data_begin.weekday()  # Sunday @ zero
        modulation = 1 - tt.abs_(tt.sin(t / 7 * np.pi + offset_rad / 2))

    if model.sim_ndim == 2:
        modulation = tt.shape_padaxis(modulation, axis=-1)

    multiplication_vec = np.ones(shape_modulation) - (1 - week_end_factor) * modulation
    new_cases_inferred_eff = new_cases_inferred * multiplication_vec
    if save_in_trace:
        pm.Deterministic("new_cases", new_cases_inferred_eff)
    return new_cases_inferred_eff


def make_change_point_RVs(
    change_points_list, pr_median_lambda_0, pr_sigma_lambda_0=1, model=None
):
    """

    Parameters
    ----------
    priors_dict
    change_points_list
    model

    Returns
    -------

    """

    default_priors_change_points = dict(
        pr_median_lambda=pr_median_lambda_0,
        pr_sigma_lambda=pr_sigma_lambda_0,
        pr_sigma_date_transient=2,
        pr_median_transient_len=4,
        pr_sigma_transient_len=0.5,
        pr_mean_date_transient=None,
    )

    for cp_priors in change_points_list:
        mh.set_missing_with_default(cp_priors, default_priors_change_points)

    model = modelcontext(model)
    len_L2 = () if model.sim_ndim == 1 else model.sim_shape[1]

    lambda_log_list = []
    tr_time_list = []
    tr_len_list = []

    #
    lambda_0_L2_log, lambda_0_L1_log = hierarchical_normal(
        "lambda_0_log",
        "sigma_lambda_0",
        np.log(pr_median_lambda_0),
        pr_sigma_lambda_0,
        len_L2,
        w=0.4,
        error_cauchy=False,
    )
    if lambda_0_L1_log is not None:
        pm.Deterministic("lambda_0_L2", tt.exp(lambda_0_L2_log))
        pm.Deterministic("lambda_0_L1", tt.exp(lambda_0_L1_log))
    else:
        pm.Deterministic("lambda_0", tt.exp(lambda_0_L2_log))

    lambda_log_list.append(lambda_0_L2_log)
    for i, cp in enumerate(change_points_list):
        lambda_cp_L2_log, lambda_cp_L1_log = hierarchical_normal(
            f"lambda_{i + 1}_log",
            f"sigma_lambda_{i + 1}",
            np.log(cp["pr_median_lambda"]),
            cp["pr_sigma_lambda"],
            len_L2,
            w=0.7,
            error_cauchy=False,
        )
        if lambda_cp_L1_log is not None:
            pm.Deterministic(f"lambda_{i + 1}_L2", tt.exp(lambda_cp_L2_log))
            pm.Deterministic(f"lambda_{i + 1}_L1", tt.exp(lambda_cp_L1_log))
        else:
            pm.Deterministic(f"lambda_{i + 1}", tt.exp(lambda_cp_L2_log))

        lambda_log_list.append(lambda_cp_L2_log)

    dt_before = model.sim_begin
    for i, cp in enumerate(change_points_list):
        dt_begin_transient = cp["pr_mean_date_transient"]
        if dt_before is not None and dt_before > dt_begin_transient:
            raise RuntimeError("Dates of change points are not temporally ordered")
        prior_mean = (dt_begin_transient - model.sim_begin).days
        tr_time_L2, _ = hierarchical_normal(
            f"transient_day_{i + 1}",
            f"sigma_transient_day_{i + 1}",
            prior_mean,
            cp["pr_sigma_date_transient"],
            len_L2,
            w=0.5,
            error_cauchy=False,
            error_fact=1.0,
        )

        tr_time_list.append(tr_time_L2)
        dt_before = dt_begin_transient

    for i, cp in enumerate(change_points_list):
        # if model.sim_ndim == 1:
        tr_len_L2_log, tr_len_L1_log = hierarchical_normal(
            f"transient_len_{i + 1}_log",
            f"sigma_transient_len_{i + 1}",
            np.log(cp["pr_median_transient_len"]),
            cp["pr_sigma_transient_len"],
            len_L2,
            w=0.7,
            error_cauchy=False,
        )
        if tr_len_L1_log is not None:
            pm.Deterministic(f"transient_len_{i + 1}_L2", tt.exp(tr_len_L2_log))
            pm.Deterministic(f"transient_len_{i + 1}_L1", tt.exp(tr_len_L1_log))
        else:
            pm.Deterministic(f"transient_len_{i + 1}", tt.exp(tr_len_L2_log))

        tr_len_list.append(tt.exp(tr_len_L2_log))
    return lambda_log_list, tr_time_list, tr_len_list


def lambda_t_with_sigmoids(
    change_points_list, pr_median_lambda_0, pr_sigma_lambda_0=0.5, model=None
):
    """

    Parameters
    ----------
    change_points_list
    pr_median_lambda_0
    pr_sigma_lambda_0
    model

    Returns
    -------

    """

    model = modelcontext(model)
    model.sim_shape = model.sim_shape

    lambda_list, tr_time_list, tr_len_list = make_change_point_RVs(
        change_points_list, pr_median_lambda_0, pr_sigma_lambda_0, model=model
    )

    # model.sim_shape = (time, state)
    # build the time-dependent spreading rate
    if len(model.sim_shape) == 2:
        lambda_t_list = [lambda_list[0] * tt.ones(model.sim_shape)]
    else:
        lambda_t_list = [lambda_list[0] * tt.ones(model.sim_shape)]
    lambda_before = lambda_list[0]

    for tr_time, tr_len, lambda_after in zip(
        tr_time_list, tr_len_list, lambda_list[1:]
    ):
        t = np.arange(model.sim_shape[0])
        tr_len = tr_len + 1e-5
        if len(model.sim_shape) == 2:
            t = np.repeat(t[:, None], model.sim_shape[1], axis=-1)
        lambda_t = tt.nnet.sigmoid((t - tr_time) / tr_len * 4) * (
            lambda_after - lambda_before
        )
        # tr_len*4 because the derivative of the sigmoid at zero is 1/4, we want to set it to 1/tr_len
        lambda_before = lambda_after
        lambda_t_list.append(lambda_t)
    lambda_t_log = sum(lambda_t_list)

    pm.Deterministic("lambda_t", tt.exp(lambda_t_log))

    return lambda_t_log


def hierarchical_normal(
    name,
    name_sigma,
    pr_mean,
    pr_sigma,
    len_L2,
    w=1.0,
    error_fact=2.0,
    error_cauchy=True,
):
    r"""
    Implements an hierarchical normal model:

    .. math::

        x_\text{L1} &= Normal(\text{pr\_mean}, \text{pr\_sigma})\\
        y_{i, \text{L2}} &= Normal(x_\text{L1}, \sigma_\text{L2})\\
        \sigma_\text{L2} &= HalfCauchy(\text{error\_fact} \cdot \text{pr\_sigma})

    It is however implemented in a non-central way, that the second line is changed to:

     .. math::

        y_{i, \text{L2}} &= x_\text{L1} +  Normal(0,1) \cdot \sigma_\text{L2}

    See for example https://arxiv.org/pdf/1312.0906.pdf


    Parameters
    ----------
    name : basestring
        Name under which :math:`x_\text{L1}` and :math:`y_\text{L2}` saved in the trace. ``'_L1'`` and ``'_L2'``
        is appended
    name_sigma : basestring
        Name under which :math:`\sigma_\text{L2}` saved in the trace. ``'_L2'`` is appended.
    pr_mean : float
        Prior mean of :math:`x_\text{L1}`
    pr_sigma : float
        Prior sigma for :math:`x_\text{L1}` and (muliplied by ``error_fact``) for :math:`\sigma_\text{L2}`
    len_L2 : int
        length of :math:`y_\text{L2}`
    error_fact : float
        Factor by which ``pr_sigma`` is multiplied as prior for `\sigma_\text{L2}`
    error_cauchy : bool
        if False, a :math:`HalfNormal` distribution is used for :math:`\sigma_\text{L2}` instead of :math:`HalfCauchy`

    Returns
    -------
    y : :class:`~theano.tensor.TensorVariable`
        the random variable :math:`y_\text{L2}`
    x : :class:`~theano.tensor.TensorVariable`
        the random variable :math:`x_\text{L1}`

    Todo
    ----
    Think about the sigma prior, whether one should model it hierarchically:
    https://projecteuclid.org/download/pdf_1/euclid.ba/1340371048 section 6

    """
    if not len_L2:  # not hierarchical
        Y = pm.Normal(name, mu=pr_mean, sigma=pr_sigma)
        X = None

    else:
        w = 1.0
        if error_cauchy:
            sigma_Y = pm.HalfCauchy(name_sigma + "_L2", beta=error_fact * pr_sigma)
        else:
            sigma_Y = pm.HalfNormal(name_sigma + "_L2", sigma=error_fact * pr_sigma)

        X = pm.Normal(name + "_L1", mu=pr_mean, sigma=pr_sigma)
        phi = pm.Normal(
            name + "_L2_raw", mu=0, sigma=1, shape=len_L2
        )  # (1-w**2)*sigma_X+1*w**2, shape=len_Y)
        Y = w * X + phi * sigma_Y
        pm.Deterministic(name + "_L2", Y)

    return Y, X


def hierarchical_beta(name, name_sigma, pr_mean, pr_sigma, len_L2):

    if not len_L2:  # not hierarchical
        Y = pm.Beta(name, alpha=pr_mean / pr_sigma, beta=1 / pr_sigma * (1 - pr_mean))
        X = None
    else:
        sigma_Y = pm.HalfCauchy(name_sigma + "_L2", beta=pr_sigma)
        X = pm.Beta(
            name + "_L1", alpha=pr_mean / pr_sigma, beta=1 / pr_sigma * (1 - pr_mean)
        )
        Y = pm.Beta(
            name + "_L2", alpha=X / sigma_Y, beta=1 / sigma_Y * (1 - X), shape=len_L2
        )

    return Y, X
