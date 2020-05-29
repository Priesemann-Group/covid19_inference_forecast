# ------------------------------------------------------------------------------ #
# Implementations of the SIR and SEIR-like models
# ------------------------------------------------------------------------------ #

import logging

import theano
import theano.tensor as tt
import numpy as np
import pymc3 as pm

from .model import *
from . import utility as ut

log = logging.getLogger(__name__)


def SIR(
    lambda_t_log,
    mu,
    name_new_I_t="new_I_t",
    name_I_begin="I_begin",
    name_I_t="I_t",
    name_S_t="S_t",
    pr_I_begin=100,
    model=None,
    return_all=False,
):
    r"""
        Implements the susceptible-infected-recovered model.

        Parameters
        ----------
        lambda_t_log : :class:`~theano.tensor.TensorVariable`
            time series of the logarithm of the spreading rate, 1 or 2-dimensional. If 2-dimensional the first
            dimension is time.
        mu : :class:`~theano.tensor.TensorVariable`
            the recovery rate :math:`\mu`, typically a random variable. Can be 0 or 1-dimensional. If 1-dimensional,
            the dimension are the different regions.
        name_new_I_t : str, optional
            Name of the ``new_I_t`` variable
        name_I_begin : str, optional
            Name of the ``I_begin`` variable
        name_I_t : str, optional
            Name of the ``I_t`` variable, set to None to avoid adding as trace variable.
        name_S_t : str, optional
            Name of the ``S_t`` variable, set to None to avoid adding as trace variable.
        pr_I_begin : float or array_like or :class:`~theano.tensor.Variable`
            Prior beta of the Half-Cauchy distribution of :math:`I(0)`.
            if type is ``tt.Constant``, I_begin will not be inferred by pymc3
        model : :class:`Cov19Model`
            if none, it is retrieved from the context
        return_all : bool
            if True, returns ``name_new_I_t``, ``name_I_t``, ``name_S_t`` otherwise returns only ``name_new_I_t``

        Returns
        ------------------
        new_I_t : :class:`~theano.tensor.TensorVariable`
            time series of the number daily newly infected persons.
        I_t : :class:`~theano.tensor.TensorVariable`
            time series of the infected (if return_all set to True)
        S_t : :class:`~theano.tensor.TensorVariable`
            time series of the susceptible (if return_all set to True)

    """
    log.info("SIR")
    model = modelcontext(model)

    # Total number of people in population
    N = model.N_population

    # Prior distributions of starting populations (infectious, susceptibles)
    if isinstance(pr_I_begin, tt.Variable):
        I_begin = pr_I_begin
    else:
        I_begin = pm.HalfCauchy(
            name=name_I_begin, beta=pr_I_begin, shape=model.shape_of_regions
        )

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
    pm.Deterministic(name_new_I_t, new_I_t)
    if name_S_t is not None:
        pm.Deterministic(name_S_t, S_t)
    if name_I_t is not None:
        pm.Deterministic(name_I_t, I_t)

    if return_all:
        return new_I_t, I_t, S_t
    else:
        return new_I_t


def SEIR(
    lambda_t_log,
    mu,
    name_new_I_t="new_I_t",
    name_new_E_t="new_E_t",
    name_I_t="I_t",
    name_S_t="S_t",
    name_I_begin="I_begin",
    name_new_E_begin="new_E_begin",
    name_median_incubation="median_incubation",
    pr_I_begin=100,
    pr_new_E_begin=50,
    pr_median_mu=1 / 8,
    pr_mean_median_incubation=4,
    pr_sigma_median_incubation=1,
    sigma_incubation=0.4,
    pr_sigma_mu=0.2,
    model=None,
    return_all=False,
):
    r"""
        Implements a model similar to the susceptible-exposed-infected-recovered model.
        Instead of a exponential decaying incubation period, the length of the period is
        lognormal distributed.

        Parameters
        ----------
        lambda_t_log : :class:`~theano.tensor.TensorVariable`
            time series of the logarithm of the spreading rate, 1 or 2-dimensional. If 2-dimensional, the first
            dimension is time.

        mu : :class:`~theano.tensor.TensorVariable`
            the recovery rate :math:`\mu`, typically a random variable. Can be 0 or
            1-dimensional. If 1-dimensional, the dimension are the different regions.

        name_new_I_t : str, optional
            Name of the ``new_I_t`` variable

        name_I_t : str, optional
            Name of the ``I_t`` variable

        name_S_t : str, optional
            Name of the ``S_t`` variable

        name_I_begin : str, optional
            Name of the ``I_begin`` variable

        name_new_E_begin : str, optional
            Name of the ``new_E_begin`` variable

        name_median_incubation : str
            The name under which the median incubation time is saved in the trace

        pr_I_begin : float or array_like
            Prior beta of the :class:`~pymc3.distributions.continuous.HalfCauchy`
            distribution of :math:`I(0)`.
            if type is ``tt.Variable``, ``I_begin`` will be set to the provided prior as
            a constant.

        pr_new_E_begin : float or array_like
            Prior beta of the :class:`~pymc3.distributions.continuous.HalfCauchy`
            distribution of :math:`E(0)`.

        pr_median_mu : float or array_like
            Prior for the median of the
            :class:`~pymc3.distributions.continuous.Lognormal` distribution of the
            recovery rate :math:`\mu`.

        pr_mean_median_incubation :
            Prior mean of the :class:`~pymc3.distributions.continuous.Normal`
            distribution of the median incubation delay  :math:`d_{\text{incubation}}`.
            Defaults to 4 days [Nishiura2020]_, which is the median serial interval (the
            important measure here is not exactly the incubation period, but the delay
            until a person becomes infectious which seems to be about 1 day earlier as
            showing symptoms).

        pr_sigma_median_incubation : number or None
            Prior sigma of the :class:`~pymc3.distributions.continuous.Normal`
            distribution of the median incubation delay  :math:`d_{\text{incubation}}`.
            If None, the incubation time will be fixed to the value of
            ``pr_mean_median_incubation`` instead of a random variable
            Default is 1 day.

        sigma_incubation :
            Scale parameter of the :class:`~pymc3.distributions.continuous.Lognormal`
            distribution of the incubation time/ delay until infectiousness. The default
            is set to 0.4, which is about the scale found in [Nishiura2020]_,
            [Lauer2020]_.

        pr_sigma_mu : float or array_like
            Prior for the sigma of the lognormal distribution of recovery rate
            :math:`\mu`.

        model : :class:`Cov19Model`
            if none, it is retrieved from the context

        return_all : bool
            if True, returns ``name_new_I_t``, ``name_new_E_t``,  ``name_I_t``,
            ``name_S_t`` otherwise returns only ``name_new_I_t``

        Returns
        -------
        name_new_I_t : :class:`~theano.tensor.TensorVariable`
            time series of the number daily newly infected persons.

        name_new_E_t : :class:`~theano.tensor.TensorVariable`
            time series of the number daily newly exposed persons. (if return_all set to
            True)

        name_I_t : :class:`~theano.tensor.TensorVariable`
            time series of the infected (if return_all set to True)

        name_S_t : :class:`~theano.tensor.TensorVariable`
            time series of the susceptible (if return_all set to True)

    """
    log.info("SEIR")
    model = modelcontext(model)

    # Build prior distrubutions:
    # --------------------------

    # Total number of people in population
    N = model.N_population

    # Prior distributions of starting populations (exposed, infectious, susceptibles)
    # We choose to consider the transitions of newly exposed people of the last 10 days.
    if isinstance(pr_new_E_begin, tt.Variable):
        new_E_begin = pr_new_E_begin
    else:
        if not model.is_hierarchical:
            new_E_begin = pm.HalfCauchy(
                name=name_new_E_begin, beta=pr_new_E_begin, shape=11
            )
        else:
            new_E_begin = pm.HalfCauchy(
                name=name_new_E_begin, beta=pr_new_E_begin, shape=(11, model.shape_of_regions)
            )

    # Prior distributions of starting populations (infectious, susceptibles)
    if isinstance(pr_I_begin, tt.Variable):
        I_begin = pr_I_begin
    else:
        I_begin = pm.HalfCauchy(
            name=name_I_begin, beta=pr_I_begin, shape=model.shape_of_regions
        )

    S_begin = N - I_begin - pm.math.sum(new_E_begin, axis=0)

    lambda_t = tt.exp(lambda_t_log)
    new_I_0 = tt.zeros_like(I_begin)

    if pr_sigma_median_incubation is None:
        median_incubation = pr_mean_median_incubation
    else:
        median_incubation = pm.Normal(
            name_median_incubation,
            mu=pr_mean_median_incubation,
            sigma=pr_sigma_median_incubation,
        )

    # Choose transition rates (E to I) according to incubation period distribution
    if not model.is_hierarchical:
        x = np.arange(1, 11)
    else:
        x = np.arange(1, 11)[:, None]

    beta = ut.tt_lognormal(x, tt.log(median_incubation), sigma_incubation)

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
    pm.Deterministic(name_new_I_t, new_I_t)

    if name_S_t is not None:
        pm.Deterministic(name_S_t, S_t)
    if name_I_t is not None:
        pm.Deterministic(name_I_t, I_t)
    if name_new_E_t is not None:
        pm.Deterministic(name_new_E_t, new_E_t)

    if return_all:
        return new_I_t, new_E_t, I_t, S_t
    else:
        return new_I_t


def uncorrelated_prior_I(
    lambda_t_log,
    mu,
    pr_median_delay,
    name_I_begin="I_begin",
    name_I_begin_ratio_log="I_begin_ratio_log",
    pr_sigma_I_begin=2,
    n_data_points_used=5,
    model=None,
):
    r"""
        Builds the prior for I begin  by solving the SIR differential from the first
        data backwards. This decorrelates the I_begin from the lambda_t at the
        beginning, allowing a more efficient sampling. The example_one_bundesland runs
        about 30\% faster with this prior, instead of a HalfCauchy.

        Parameters
        ----------
        lambda_t_log : TYPE
            Description
        mu : TYPE
            Description
        pr_median_delay : TYPE
            Description
        name_I_begin : str, optional
            Description
        name_I_begin_ratio_log : str, optional
            Description
        pr_sigma_I_begin : int, optional
            Description
        n_data_points_used : int, optional
            Description
        model : :class:`Cov19Model`
            if none, it is retrieved from the context
        lambda_t_log : :class:`~theano.tensor.TensorVariable`
        mu : :class:`~theano.tensor.TensorVariable`
        pr_median_delay : float
        pr_sigma_I_begin : float
        n_data_points_used : int

        Returns
        ------------------
        I_begin: :class:`~theano.tensor.TensorVariable`

    """
    log.info("Uncorrelated prior_I")
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
            name=name_I_begin_ratio_log, mu=0, sigma=pr_sigma_I_begin, shape=num_regions
        )
    )
    pm.Deterministic(name_I_begin, I_begin)
    return I_begin
