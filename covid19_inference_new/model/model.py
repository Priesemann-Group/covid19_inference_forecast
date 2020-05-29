# ------------------------------------------------------------------------------ #
# This file implements the abstract model base class.
# It has some helper properties to access date ranges and hierarchical details
# ------------------------------------------------------------------------------ #

import datetime
import logging

import numpy as np
from pymc3 import Model  # this import is needed to get pymc3-style "with ... as model:"

# we cannot import utility, would create recursive dependencies
# from . import utility as ut

log = logging.getLogger(__name__)


# can we rename this guy to model base or something?
class Cov19Model(Model):
    """
        Abstract base class for the dynamic model of covid-19 propagation.
        Derived from :class:`pymc3.Model`.

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
    def shape_of_regions(self):
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


def set_missing_priors_with_default(priors_dict, default_priors):
    """
        Takes a dict with custom priors and a dict with defaults and sets keys that
        are not given
    """
    for prior_name in priors_dict.keys():
        if prior_name not in default_priors:
            log.warning(f"Prior with name {prior_name} not known")

    for prior_name, value in default_priors.items():
        if prior_name not in priors_dict:
            priors_dict[prior_name] = value
            log.info(f"{prior_name} was set to default value {value}")
