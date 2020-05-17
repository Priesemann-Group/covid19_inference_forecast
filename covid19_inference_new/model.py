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

if platform.system() == "Darwin":
    theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"  # workaround for macos


class Cov19Model(Model):
    """
        Model class used to create a covid-19 propagation dynamics model.
        Parameters below are passed to the constructor.
        Attributes (Variables) are available after creation and can be accessed from
        every instance. Some background:

            * The simulation starts `diff_data_sim` days before the data.
            * The data has a certain length, on which the inference is based. This
              length is given by `new_cases_obs`.
            * After the inference, a forecast takes of length `fcast_len` takes
              place, starting on the day after the last data point in `new_cases_obs`.
            * In total, traces produced by a model run have the length
              `sim_len = diff_data_sim + data_len + fcast_len`
            * Date ranges include both boundaries. For example, if `data_begin` is March
              1 and `data_end` is March 3 then `data_len` will be 3.

        Parameters
        ----------
        new_cases_obs : 1 or 2d array
            If the array is two-dimensional, an hierarchical model will be constructed.
            First dimension is then time, the second the region/country.
        data_begin : datatime.datetime
            Date of the first data point
        fcast_len : int
            Number of days the simulations runs longer than the data
        diff_data_sim : int
            Number of days the simulation starts earlier than the data. Should be
            significantly longer than the delay between infection and report of cases.
        N_population : number or 1d array
            Number of inhabitance in region, needed for the S(E)IR model. Is ideally 1
            dimensional if new_cases_obs is 2 dimensional
        name : string
            suffix appended to the name of random variables saved in the trace
        model :
            specify a model, if this one should expand another

        Attributes
        ----------
        new_cases_obs : 1 or 2d array
            as passed during construction

        data_begin : datatime.datetime
            date of the first data point in the data

        data_end : datatime.datetime
            date of the last data point in the data

        sim_begin : datatime.datetime
            date at which the simulation begins

        sim_end : datatime.datetime
            date at which the simulation ends (should match fcast_end)

        fcast_begin : datatime.datetime
            date at which the forecast starts (should be one day after data_end)

        fcast_end : datatime.datetime
            data at which the forecast ends

        data_len : int
            total number of days in the data

        sim_len : int
            total number of days in the simulation

        fcast_len : int
            total number of days in the forecast

        diff_data_sim : int
            difference in days between the simulation begin and the data begin.
            The simulation starting time is usually earlier than the data begin.

        Example
        -------
        .. code-block::

            with Cov19Model(**params) as model:
                # Define model here
    """

    def __init__(
        self,
        new_cases_obs,
        data_begin,
        fcast_len,
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

        # assign properties
        self._data_begin = data_begin
        self._sim_begin = self.data_begin - datetime.timedelta(days=diff_data_sim)
        self._data_end = self.data_begin + datetime.timedelta(
            days=len(new_cases_obs) - 1
        )
        self._sim_end = self.data_end + datetime.timedelta(days=fcast_len)

        # totel length of simulation, get later via the shape
        sim_len = len(new_cases_obs) + diff_data_sim + fcast_len
        if sim_len < len(new_cases_obs) + diff_data_sim:
            raise RuntimeError(
                "Simulation ends before the end of the data. Increase num_days_sim."
            )

        # shape and dimension of simulation
        if self.sim_ndim == 1:
            self.sim_shape = (sim_len,)
        elif self.sim_ndim == 2:
            self.sim_shape = (sim_len, self.new_cases_obs.shape[1])

        if self.data_end > datetime.datetime.today():
            log.warning(
                f"Your last data point is in the future ({self.data_end}). "
                + "Are you traveling faster than light?"
            )

    """
        Properties
        ----------
        Useful properties, mainly used by the plot module.
    """

    """
        Utility properties
    """

    @property
    def shape_num_regions(self):
        # Number of regions as tuple of int
        return () if self.sim_ndim == 1 else self.sim_shape[1]

    @property
    def is_hierarchical(self):
        return self.new_cases_obs.ndim == 2

    """
        Forecast properties
    """

    @property
    def fcast_begin(self):
        # Returns date on which the forecast starts i.e. the day after the data ends
        return self.data_end + datetime.timedelta(days=1)

    @property
    def fcast_end(self):
        # Returns date on which the simulation and the forecast end
        return self.sim_end

    @property
    def fcast_len(self):
        # Returns the length of the forecast in days
        return (self.sim_end - self.data_end).days

    """
        Data properties
    """

    @property
    def data_len(self):
        return self.new_cases_obs.shape[0]

    @property
    def data_dim(self):
        return self.new_cases_obs.shape[1]

    @property
    def data_begin(self):
        return self._data_begin

    @property
    def data_end(self):
        return self._data_end

    """
        Simulation properties
    """

    @property
    def sim_len(self):
        return self.sim_shape[0]

    @property
    def sim_begin(self):
        return self._sim_begin

    @property
    def sim_end(self):
        return self._sim_end

    @property
    def diff_data_sim(self):
        return (self.data_begin - self.sim_begin).days


def modelcontext(model):
    """
        return the given model or try to find it in the context if there was
        none supplied.
    """
    if model is None:
        return Cov19Model.get_context()
    return model


def student_t_likelihood(
    new_cases_inferred,
    pr_beta_sigma_obs=30,
    nu=4,
    offset_sigma=1,
    model=None,
    data_obs=None,
    name_student_t="_new_cases_studentT",
    name_sigma_obs="sigma_obs",
):
    r"""
        Set the likelihood to apply to the model observations (`model.new_cases_obs`)
        We assume a :class:`~pymc3.distributions.continuous.StudentT` distribution because it is robust against outliers [Lange1989]_.
        The likelihood follows:

        .. math::

            P(\text{data\_obs}) &\sim StudentT(\text{mu} = \text{new\_cases\_inferred}, sigma =\sigma,
            \text{nu} = \text{nu})\\
            \sigma &= \sigma_r \sqrt{\text{new\_cases\_inferred} + \text{offset\_sigma}}

        The parameter :math:`\sigma_r` follows
        a :class:`~pymc3.distributions.continuous.HalfCauchy` prior distribution with parameter beta set by
        ``pr_beta_sigma_obs``. If the input is 2 dimensional, the parameter :math:`\sigma_r` is different for every region.

        Parameters
        ----------
        new_cases_inferred : :class:`~theano.tensor.TensorVariable`
            One or two dimensonal array. If 2 dimensional, the first dimension is time and the second are the
            regions/countries

        pr_beta_sigma_obs : float
            The beta of the :class:`~pymc3.distributions.continuous.HalfCauchy` prior distribution of :math:`\sigma_r`.

        nu : float
            How flat the tail of the distribution is. Larger nu should  make the model
            more robust to outliers. Defaults to 4 [Lange1989]_.

        offset_sigma : float
            An offset added to the sigma, to make the inference procedure robust. Otherwise numbers of
            ``new_cases_inferred`` would lead to very small errors and diverging likelihoods. Defaults to 1.

        model:
            The model on which we want to add the distribution

        data_obs : array
            The data that is observed. By default it is ``model.new_cases_ob``

        name_student_t :
            The name under which the studentT distribution is saved in the trace.

        name_sigma_obs :
            The name under which the distribution of the observable error is saved in the trace

        Returns
        -------
        None

        References
        ----------

        .. [Lange1989] Lange, K., Roderick J. A. Little, & Jeremy M. G. Taylor. (1989).
            Robust Statistical Modeling Using the t Distribution.
            Journal of the American Statistical Association,
            84(408), 881-896. doi:10.2307/2290063

    """

    model = modelcontext(model)

    len_sigma_obs = () if model.sim_ndim == 1 else model.sim_shape[1]
    sigma_obs = pm.HalfCauchy(
        name_sigma_obs, beta=pr_beta_sigma_obs, shape=len_sigma_obs
    )

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
    lambda_t_log, mu, pr_I_begin=100, model=None, return_all=False, save_all=False,
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

        mu : :class:`~theano.tensor.TensorVariable`
            the recovery rate :math:`\mu`, typically a random variable. Can be 0 or 1-dimensional. If 1-dimensional,
            the dimension are the different regions.

        pr_I_begin : float or array_like or :class:`~theano.tensor.TensorVariable`
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

        new_I_t : :class:`~theano.tensor.TensorVariable`
            time series of the number daily newly infected persons.
        I_t : :class:`~theano.tensor.TensorVariable`
            time series of the infected (if return_all set to True)
        S_t : :class:`~theano.tensor.TensorVariable`
            time series of the susceptible (if return_all set to True)

    """
    model = modelcontext(model)

    # Total number of people in population
    N = model.N_population

    # Number of regions as tuple of int
    num_regions = () if model.sim_ndim == 1 else model.sim_shape[1]

    # Prior distributions of starting populations (infectious, susceptibles)
    if isinstance(pr_I_begin, tt.TensorVariable):
        I_begin = pr_I_begin
    else:
        I_begin = pm.HalfCauchy(name="I_begin", beta=pr_I_begin, shape=num_regions)

    S_begin = N - I_begin

    lambda_t = tt.exp(lambda_t_log)
    new_I_0 = tt.zeros_like(I_begin)

    # Runs SIR model:
    def next_day(lambda_t, S_t, I_t, _, mu, N):
        new_I_t = lambda_t / N * I_t * S_t
        S_t = S_t - new_I_t
        I_t = I_t + new_I_t - mu * I_t
        I_t = tt.clip(I_t, -1, N)  # for stability
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
    pr_mean_median_incubation=4,
    pr_sigma_median_incubation=1,
    sigma_incubation=0.4,
    pr_sigma_mu=0.2,
    model=None,
    return_all=False,
    save_all=False,
    name_median_incubation="median_incubation",
):
    r"""
        Implements a model similar to the susceptible-exposed-infected-recovered model. Instead of a exponential decaying
        incubation period, the length of the period is lognormal distributed. The complete equation is:

         .. math::

            E_{\text{new}}(t) &= \lambda_t I(t-1) \frac{S(t)}{N}   \\
            S(t) &= S(t-1) - E_{\text{new}}(t)  \\
            I_\text{new}(t) &= \sum_{k=1}^{10} \beta(k) E_{\text{new}}(t-k)   \\
            I(t) &= I(t-1) + I_{\text{new}}(t) - \mu  I(t) \\
            \beta(k) & = P(k) \sim LogNormal(\text{log}(d_{\text{incubation}})), \text{sigma\_incubation})

        The recovery rate :math:`\mu` and the incubation period is the same for all regions and follow respectively:

        .. math::

             P(\mu) &\sim LogNormal(\text{log(pr\_median\_mu)), pr\_sigma\_mu}) \\
             P(d_{\text{incubation}}) &\sim Normal(\text{pr\_mean\_median\_incubation, pr\_sigma\_median\_incubation})

        The initial number of infected and newly exposed differ for each region and follow prior
        :class:`~pymc3.distributions.continuous.HalfCauchy` distributions:

        .. math::

             E(t)  &\sim HalfCauchy(\text{pr\_beta\_E\_begin}) \:\: \text{ for} \: t \in \{-9, -8, ..., 0\}\\
             I(0)  &\sim HalfCauchy(\text{pr\_beta\_I\_begin}).


        Parameters
        ----------
        lambda_t_log : :class:`~theano.tensor.TensorVariable`
            time series of the logarithm of the spreading rate, 1 or 2-dimensional. If 2-dimensional, the first
            dimension is time.
        pr_beta_I_begin : float or array_like
            Prior beta of the :class:`~pymc3.distributions.continuous.HalfCauchy` distribution of :math:`I(0)`.
        pr_beta_new_E_begin : float or array_like
            Prior beta of the :class:`~pymc3.distributions.continuous.HalfCauchy` distribution of :math:`E(0)`.
        pr_median_mu : float or array_like
            Prior for the median of the :class:`~pymc3.distributions.continuous.Lognormal` distribution of the recovery rate :math:`\mu`.
        pr_mean_median_incubation :
            Prior mean of the :class:`~pymc3.distributions.continuous.Normal` distribution of the median incubation delay  :math:`d_{\text{incubation}}`.
            Defaults to 4 days [Nishiura2020]_, which is the median serial interval (the important measure here is not exactly
            the incubation period, but the delay until a person becomes infectious which seems to be about
            1 day earlier as showing symptoms).
        pr_sigma_median_incubation :
            Prior sigma of the :class:`~pymc3.distributions.continuous.Normal` distribution of the median incubation delay  :math:`d_{\text{incubation}}`.
            Default is 1 day.
        sigma_incubation :
            Scale parameter of the :class:`~pymc3.distributions.continuous.Lognormal` distribution of the incubation time/
            delay until infectiousness. The default is set to 0.4, which is about the scale found in [Nishiura2020]_, [Lauer2020]_.
        pr_sigma_mu : float or array_like
            Prior for the sigma of the lognormal distribution of recovery rate :math:`\mu`.
        model : :class:`Cov19Model`
          if none, it is retrieved from the context
        return_all : bool
            if True, returns ``new_I_t``, ``new_E_t``,  ``I_t``, ``S_t`` otherwise returns only ``new_I_t``
        save_all : bool
            if True, saves ``new_I_t``, ``new_E_t``, ``I_t``, ``S_t`` in the trace, otherwise it saves only ``new_I_t``
        name_median_incubation : str
            The name under which the median incubation time is saved in the trace

        Returns
        -------

        new_I_t : :class:`~theano.tensor.TensorVariable`
            time series of the number daily newly infected persons.
        new_E_t : :class:`~theano.tensor.TensorVariable`
            time series of the number daily newly exposed persons. (if return_all set to True)
        I_t : :class:`~theano.tensor.TensorVariable`
            time series of the infected (if return_all set to True)
        S_t : :class:`~theano.tensor.TensorVariable`
            time series of the susceptible (if return_all set to True)

        References
        ----------

        .. [Nishiura2020] Nishiura, H.; Linton, N. M.; Akhmetzhanov, A. R.
            Serial Interval of Novel Coronavirus (COVID-19) Infections.
            Int. J. Infect. Dis. 2020, 93, 284–286. https://doi.org/10.1016/j.ijid.2020.02.060.
        .. [Lauer2020] Lauer, S. A.; Grantz, K. H.; Bi, Q.; Jones, F. K.; Zheng, Q.; Meredith, H. R.; Azman, A. S.; Reich, N. G.; Lessler, J.
            The Incubation Period of Coronavirus Disease 2019 (COVID-19) From Publicly Reported Confirmed Cases: Estimation and Application.
            Ann Intern Med 2020. https://doi.org/10.7326/M20-0504.


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
    # We choose to consider the transitions of newly exposed people of the last 10 days.
    if num_regions == ():
        new_E_begin = pm.HalfCauchy(
            name="new_E_begin", beta=pr_beta_new_E_begin, shape=11
        )
    else:
        new_E_begin = pm.HalfCauchy(
            name="new_E_begin", beta=pr_beta_new_E_begin, shape=(11, num_regions)
        )
    I_begin = pm.HalfCauchy(name="I_begin", beta=pr_beta_I_begin, shape=num_regions)
    S_begin = N - I_begin - pm.math.sum(new_E_begin, axis=0)

    lambda_t = tt.exp(lambda_t_log)
    new_I_0 = tt.zeros_like(I_begin)

    median_incubation = pm.Normal(
        name_median_incubation,
        mu=pr_mean_median_incubation,
        sigma=pr_sigma_median_incubation,
    )

    # Choose transition rates (E to I) according to incubation period distribution
    if not num_regions:
        x = np.arange(1, 11)
    else:
        x = np.arange(1, 11)[:, None]

    beta = mh.tt_lognormal(x, tt.log(median_incubation), sigma_incubation)

    # Runs SEIR model:
    def next_day(
        lambda_t,
        S_t,
        nE1,
        nE2,
        nE3,
        nE4,
        nE5,
        nE6,
        nE7,
        nE8,
        nE9,
        nE10,
        I_t,
        _,
        mu,
        beta,
        N,
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
        I_t = tt.clip(I_t, -1, N - 1)  # for stability
        S_t = tt.clip(S_t, -1, N)
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
    S_t, new_E_t, I_t, new_I_t = outputs
    pm.Deterministic("new_I_t", new_I_t)
    if save_all:
        pm.Deterministic("S_t", S_t)
        pm.Deterministic("I_t", I_t)
        pm.Deterministic("new_E_t", new_E_t)

    if return_all:
        return new_I_t, new_E_t, I_t, S_t
    else:
        return new_I_t


def delay_cases(
    new_I_t,
    pr_median_delay=10,
    pr_sigma_median_delay=0.2,
    pr_median_scale_delay=0.3,
    pr_sigma_scale_delay=None,
    model=None,
    save_in_trace=True,
    name_delay="delay",
    name_delayed_cases="new_cases_raw",
    len_input_arr=None,
    len_output_arr=None,
    diff_input_output=None,
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
            The mean of the :class:`~pymc3.distributions.continuous.normal` distribution which
            models the prior median of the :class:`~pymc3.distributions.continuous.LogNormal` delay kernel.
        pr_sigma_median_delay : float
            The standart devaiation of :class:`~pymc3.distributions.continuous.normal` distribution which
            models the prior median of the :class:`~pymc3.distributions.continuous.LogNormal` delay kernel.
        pr_median_scale_delay : float
            The scale (width) of the :class:`~pymc3.distributions.continuous.LogNormal` delay kernel.
        pr_sigma_scale_delay : float
            If it is not None, the scale is of the delay is kernel follows a prior
            :class:`~pymc3.distributions.continuous.LogNormal` distribution, with median ``pr_median_scale_delay`` and
            scale ``pr_sigma_scale_delay``.
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
        len_input_arr :
            Length of ``new_I_t``. By default equal to ``model.sim_len``. Necessary because the shape of theano
            tensors are not defined at when the graph is built.
        len_output_arr : int
            Length of the array returned. By default it set to the length of the cases_obs saved in the model plus
            the number of days of the forecast.
        diff_input_output : int
            Number of days the returned array begins later then the input. Should be significantly larger than
            the median delay. By default it is set to the ``model.diff_data_sim``.

        Returns
        -------
        new_cases_inferred : :class:`~theano.tensor.TensorVariable`
            The delayed input :math:`y_\text{delayed}(t)`, typically the daily number new cases that one expects to measure.
    """

    model = modelcontext(model)

    if len_output_arr is None:
        len_output_arr = model.data_len + model.fcast_len
    if diff_input_output is None:
        diff_input_output = model.diff_data_sim
    if len_input_arr is None:
        len_input_arr = model.sim_len

    len_delay = () if model.sim_ndim == 1 else model.sim_shape[1]
    delay_L2_log, delay_L1_log = hierarchical_normal(
        name_delay + "_log",
        "sigma_" + name_delay,
        np.log(pr_median_delay),
        pr_sigma_median_delay,
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
        len_input_arr=len_input_arr,
        len_output_arr=len_output_arr,
        median_delay=tt.exp(delay_L2_log),
        scale_delay=tt.exp(scale_delay_L2_log),
        delay_betw_input_output=diff_input_output,
    )
    if save_in_trace:
        pm.Deterministic(name_delayed_cases, new_cases_inferred)

    return new_cases_inferred


def week_modulation(
    new_cases_raw,
    week_modulation_type="abs_sine",
    pr_mean_weekend_factor=0.3,
    pr_sigma_weekend_factor=0.5,
    week_end_days=(6, 7),
    model=None,
    save_in_trace=True,
):
    r"""
    Adds a weekly modulation of the number of new cases:

    .. math::
        \text{new\_cases} &= \text{new\_cases\_raw} \cdot (1-f(t))\,, \qquad\text{with}\\
        f(t) &= f_w \cdot \left(1 - \left|\sin\left(\frac{\pi}{7} t- \frac{1}{2}\Phi_w\right)\right| \right),

    if ``week_modulation_type`` is ``"abs_sine"`` (the default). If ``week_modulation_type`` is ``"step"``, the
    new cases are simply multiplied by the weekend factor on the days set by ``week_end_days``

    The weekend factor :math:`f_w` follows a Lognormal distribution with
    median ``pr_mean_weekend_factor`` and sigma ``pr_sigma_weekend_factor``. It is hierarchically constructed if
    the input is two-dimensional by the function :func:`hierarchical_normal` with default arguments.

    The offset from Sunday :math:`\Phi_w` follows a flat :class:`~pymc3.distributions.continuous.VonMises` distribution
    and is the same for all regions.

    Parameters
    ----------

    new_cases_raw : :class:`~theano.tensor.TensorVariable`
        The input array, can be one- or two-dimensional
    week_modulation_type : str
        The type of modulation, accepts ``"step"`` or  ``"abs_sine`` (the default).
    pr_mean_weekend_factor : float
        Sets the prior mean of the factor :math:`f_w` by which weekends are counted.
    pr_sigma_weekend_factor : float
        Sets the prior sigma of the factor :math:`f_w` by which weekends are counted.
    week_end_days : tuple of ints
        The days counted as weekend if ``week_modulation_type`` is ``"step"``
    model : :class:`Cov19Model`
        if none, it is retrieved from the context
    save_in_trace : bool
        If True (default) the new_cases are saved in the trace.

    Returns
    -------

    new_cases : :class:`~theano.tensor.TensorVariable`

    """
    model = modelcontext(model)
    shape_modulation = list(model.sim_shape)
    shape_modulation[0] -= model.diff_data_sim

    len_L2 = () if model.sim_ndim == 1 else model.sim_shape[1]

    week_end_factor_L2_log, week_end_factor_L1_log = hierarchical_normal(
        "weekend_factor_log",
        "sigma_weekend_factor",
        pr_mean=tt.log(pr_mean_weekend_factor),
        pr_sigma=pr_sigma_weekend_factor,
        len_L2=len_L2,
    )

    week_end_factor_L2 = tt.exp(week_end_factor_L2_log)

    if model.sim_ndim == 1:
        pm.Deterministic("weekend_factor", week_end_factor_L2)
    elif model.sim_ndim == 2:
        week_end_factor_L1 = tt.exp(week_end_factor_L1_log)
        pm.Deterministic("weekend_factor_L2", week_end_factor_L2)
        pm.Deterministic("weekend_factor_L1", week_end_factor_L1)

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

    multiplication_vec = tt.abs_(
        np.ones(shape_modulation) - week_end_factor_L2 * modulation
    )
    new_cases_inferred_eff = new_cases_raw * multiplication_vec
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
        relative_to_previous=False,
        pr_factor_to_previous=1,
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
        if cp["relative_to_previous"]:
            pr_sigma_lambda = lambda_log_list[-1] + tt.log(cp["pr_factor_to_previous"])
        else:
            pr_sigma_lambda = np.log(cp["pr_median_lambda"])
        lambda_cp_L2_log, lambda_cp_L1_log = hierarchical_normal(
            f"lambda_{i + 1}_log",
            f"sigma_lambda_{i + 1}",
            pr_sigma_lambda,
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
    model : :class:`Cov19Model`
        if none, it is retrieved from the context

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

    It is however implemented in a non-centered way, that the second line is changed to:

     .. math::

        y_{i, \text{L2}} &= x_\text{L1} +  Normal(0,1) \cdot \sigma_\text{L2}

    See for example https://arxiv.org/pdf/1312.0906.pdf


    Parameters
    ----------
    name : str
        Name under which :math:`x_\text{L1}` and :math:`y_\text{L2}` saved in the trace. ``'_L1'`` and ``'_L2'``
        is appended
    name_sigma : str
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


def make_prior_I(
    lambda_t_log,
    mu,
    pr_median_delay,
    pr_sigma_I_begin=2,
    n_data_points_used=5,
    model=None,
):
    """
    Builds the prior for I begin  by solving the SIR differential from the first data backwards. This decorrelates the
    I_begin from the lambda_t at the beginning, allowing a more efficient sampling. The example_one_bundesland runs
    about 30\% faster with this prior, instead of a HalfCauchy.

    Parameters
    ----------
    lambda_t_log : :class:`~theano.tensor.TensorVariable`
    mu : :class:`~theano.tensor.TensorVariable`
    pr_median_delay : float
    pr_sigma_I_begin : float
    n_data_points_used : int
    model : :class:`Cov19Model`
        if none, it is retrieved from the context

    Returns
    -------
    I_begin: :class:`~theano.tensor.TensorVariable`

    """
    model = modelcontext(model)

    num_regions = () if model.sim_ndim == 1 else model.sim_shape[1]

    lambda_t = tt.exp(lambda_t_log)

    delay = round(pr_median_delay)
    num_new_I_ref = np.mean(model.new_cases_obs[:n_data_points_used], axis=0)
    days_diff = model.diff_data_sim - delay + 3
    I_ref = num_new_I_ref / lambda_t[days_diff]
    I0_ref = I_ref / (1 + lambda_t[days_diff // 2] - mu) ** days_diff
    I_begin = I0_ref * tt.exp(
        pm.Normal(
            name="I_begin_ratio_log", mu=0, sigma=pr_sigma_I_begin, shape=num_regions
        )
    )
    pm.Deterministic("I_begin", I_begin)
    return I_begin


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
