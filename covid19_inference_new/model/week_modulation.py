# ------------------------------------------------------------------------------ #
# Apply a weekly modulation to the reported cases. Less reports on the weekend
# ------------------------------------------------------------------------------ #

import logging

import theano
import theano.tensor as tt
import numpy as np
import pymc3 as pm

from .model import *
from . import utility as ut

log = logging.getLogger(__name__)


def week_modulation(
    cases,
    name_cases=None,
    name_weekend_factor="weekend_factor",
    name_offset_modulation="offset_modulation",
    week_modulation_type="abs_sine",
    pr_mean_weekend_factor=0.3,
    pr_sigma_weekend_factor=0.5,
    weekend_days=(6, 7),
    model=None,
):
    r"""
    Adds a weekly modulation of the number of new cases:

    .. math::
        \text{new\_cases} &= \text{new\_cases\_raw} \cdot (1-f(t))\,, \qquad\text{with}\\
        f(t) &= f_w \cdot \left(1 - \left|\sin\left(\frac{\pi}{7} t- \frac{1}{2}\Phi_w\right)\right| \right),

    if ``week_modulation_type`` is ``"abs_sine"`` (the default). If ``week_modulation_type`` is ``"step"``, the
    new cases are simply multiplied by the weekend factor on the days set by ``weekend_days``

    The weekend factor :math:`f_w` follows a Lognormal distribution with
    median ``pr_mean_weekend_factor`` and sigma ``pr_sigma_weekend_factor``. It is hierarchically constructed if
    the input is two-dimensional by the function :func:`hierarchical_normal` with default arguments.

    The offset from Sunday :math:`\Phi_w` follows a flat :class:`~pymc3.distributions.continuous.VonMises` distribution
    and is the same for all regions.

    Parameters
    ----------

    cases : :class:`~theano.tensor.TensorVariable`
        The input array of daily new cases, can be one- or two-dimensional
    name_cases : str or None,
        The name under which to save the cases as a trace variable.
        Default: None, cases are not stored in the trace.
    week_modulation_type : str
        The type of modulation, accepts ``"step"`` or  ``"abs_sine`` (the default).
    pr_mean_weekend_factor : float
        Sets the prior mean of the factor :math:`f_w` by which weekends are counted.
    pr_sigma_weekend_factor : float
        Sets the prior sigma of the factor :math:`f_w` by which weekends are counted.
    weekend_days : tuple of ints
        The days counted as weekend if ``week_modulation_type`` is ``"step"``
    model : :class:`Cov19Model`
        if none, it is retrieved from the context

    Returns
    -------

    new_cases : :class:`~theano.tensor.TensorVariable`

    """

    def step_modulation():
        """
        Helper function for the step modulation

        Returns
        -------
        modulation
        """
        modulation = np.zeros(shape_modulation[0])
        for i in range(shape_modulation[0]):
            date_curr = model.data_begin + datetime.timedelta(days=i)
            if date_curr.isoweekday() in weekend_days:
                modulation[i] = 1
        return modulation

    def abs_sine_modulation():
        """
        Helper function for the absolute sin modulation

        Returns
        -------
        modulation
        """
        offset_rad = pm.VonMises(name_offset_modulation + "_rad", mu=0, kappa=0.01)
        offset = pm.Deterministic(name_offset_modulation, offset_rad / (2 * np.pi) * 7)
        t = np.arange(shape_modulation[0]) - model.data_begin.weekday()  # Sunday @ zero
        modulation = 1 - tt.abs_(tt.sin(t / 7 * np.pi + offset_rad / 2))
        return modulation

    log.info("Week modulation")
    # Create our model context
    model = modelcontext(model)

    # Get the shape of the modulation from the shape of our simulation
    shape_modulation = list(model.sim_shape)
    #shape_modulation[0] -= model.diff_data_sim

    if not model.is_hierarchical:
        weekend_factor_log = pm.Normal(
            name=name_weekend_factor + "_log",
            mu=tt.log(pr_mean_weekend_factor),
            sigma=pr_sigma_weekend_factor,
        )
        weekend_factor = tt.exp(weekend_factor_log)
        pm.Deterministic(name_weekend_factor, weekend_factor)

    else:  # hierarchical
        weekend_factor_L2_log, weekend_factor_L1_log = ut.hierarchical_normal(
            name_L1=name_weekend_factor + "_hc_L1_log",
            name_L2=name_weekend_factor + "_hc_L2_log",
            name_sigma="sigma_" + name_weekend_factor,
            pr_mean=tt.log(pr_mean_weekend_factor),
            pr_sigma=pr_sigma_weekend_factor,
        )

        # We do that so we can use it later (same name as non hierarchical)
        weekend_factor_L1 = tt.exp(weekend_factor_L1_log)
        weekend_factor_L2 = tt.exp(weekend_factor_L2_log)
        pm.Deterministic(name_weekend_factor + "_hc_L1", weekend_factor_L1)
        pm.Deterministic(name_weekend_factor + "_hc_L2", weekend_factor_L2)
        weekend_factor = weekend_factor_L2

    # Different modulation types
    modulation = step_modulation() if week_modulation_type == "step" else 0
    modulation = abs_sine_modulation() if week_modulation_type == "abs_sine" else 0

    if model.is_hierarchical:
        modulation = tt.shape_padaxis(modulation, axis=-1)

    multiplication_vec = tt.abs_(
        np.ones(shape_modulation) - weekend_factor * modulation
    )

    new_cases_inferred_eff = cases * multiplication_vec

    if name_cases is not None:
        pm.Deterministic(name_cases, new_cases_inferred_eff)

    return new_cases_inferred_eff
