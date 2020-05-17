__version__ = "unknown"
from ._version import __version__

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s [%(name)s] %(message)s")
log = logging.getLogger(__name__)
from . import plotting

# from .data_retrieval import GOOGLE
from . import data_retrieval
from . import plot
from .model import (
    Cov19Model,
    lambda_t_with_sigmoids,
    SIR,
    delay_cases,
    week_modulation,
    student_t_likelihood,
    make_prior_I,
)
from ._dev_helper import create_example_instance
