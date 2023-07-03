"""Solvers for ODEs.

Extends the existing classes in scipy.integrate.
"""

import numpy as np
import scipy.integrate
import warnings


class FDDenseOutput(scipy.integrate._ivp.base.ConstantDenseOutput):
    #     #
    pass


class ForwardEuler(scipy.integrate.OdeSolver):
    """Explicit Forward Euler solver.

    Uses a fixed, uniform step size.
    """
    def __init__(self,
                 fun,
                 t0,
                 y0,
                 t_bound,
                 step_size=0.001,
                 vectorized=False,
                 **extraneous):

        if extraneous:
            warnings.warn('The following arguments have no effect for '
                          'Forward Euler solver: {}.'
                          .format(', '.join('`{}`'.format(x)
                                            for x in extraneous)))

        super(ForwardEuler, self).__init__(
            fun,
            t0,
            y0,
            t_bound,
            vectorized,
            support_complex=True)

        self.fixed_step_size = step_size

    def _step_impl(self):
        self.y_old = self.y.copy()

        self.y += self.fun(self.t, self.y) * self.fixed_step_size
        self.t += self.fixed_step_size

        if np.any(np.isinf(self.y)):
            return False, 'infinite y'

        return True, None

    def _dense_output_impl(self):
        return FDDenseOutput(self.t_old, self.t, self.y_old)
