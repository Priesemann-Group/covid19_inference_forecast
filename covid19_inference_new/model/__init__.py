from .model import Cov19Model
from .compartmental_models import SIR, SEIR, uncorrelated_prior_I
from .delay import delay_cases
from .spreading_rate import lambda_t_with_sigmoids
from .likelihood import student_t_likelihood
from .week_modulation import week_modulation

# make everything available but hidden
from . import utility as _utility
from . import model as _model
from . import compartmental_models as _compartmental_models
from . import delay as _delay
from . import spreading_rate as _spreading_rate
from . import likelihood as _likelihood
from . import week_modulation as _week_modulation

"""
# TODO 2020-05-19

* pip?
* automated ipynb execution new repo
    + script that converts and runs and creates badge
* automated figure creation old repo


# Aim
* name(s) argument with defaults
    * explicit name_I_t = "I_t"
    * if name_I_t = None, do not export (for the optional guys)
    * _hc_L1 to indicate hierarchical variables
        + if hc, then we append suffixes to provided argument names
    * last suffix `_log_` vs `_log__` from pymc3
    * last suffix `_raw_` to indicate things that need more cleanup
* model as optional argument
* _try_ to separate hc from non hc
* comments within functions what;s happening



# Delegate
* [x] SM __init__.py
* [x] PS model.py
* [x] JPN compartmental_models.py
* [x] SM spreading_rate.py
* [x] JPN week_modulation.py
* [x] PS delay.py
* [x] SM likelihood.py
* [x] JPN utility.py

* [ ] JD testing

* [ ] fix examples
* [ ] ipynb automation
* [ ] figure automation
* [ ] fix documentation
* [ ] add details on the naming convention to the doc.
"""
