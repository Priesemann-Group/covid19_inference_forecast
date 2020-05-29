import datetime
import pandas as pd
import logging

# Import base class
from .retrieval import Retrieval, _data_dir_fallback

log = logging.getLogger(__name__)


class OWD(Retrieval):
    """
    This class can be used to retrieve the testings dataset from
    `Our World in Data <https://ourworldindata.org/coronavirus>`_.

    Example
    -------
    .. code-block::

        owd = cov19.data_retrieval.OWD()
        owd.download_all_available_data()

    """

    def __init__(self, auto_download=False):
        """
        On init of this class the base Retrieval Class __init__ is called, with google specific
        arguments.

        Parameters
        ----------
        auto_download : bool, optional
            Whether or not to automatically call the download_all_available_data() method.
            One should explicitly call this method for more configuration options
            (default: false)
        """

        # ------------------------------------------------------------------------------ #
        #  Init Retrieval Base Class
        # ------------------------------------------------------------------------------ #
        """
        A name mainly used for the Local Filename
        """
        name = "OurWorldinData"

        """
        The url to the main dataset as csv, if none if supplied the fallback routines get used
        """
        url_csv = "https://covid.ourworldindata.org/data/owid-covid-data.csv"

        """
        Kwargs for pandas read csv
        """
        kwargs = {}  # Surpress warning

        """
        If the local file is older than the update_interval it gets updated once the
        download all function is called. Can be diffent values depending on the parent class
        """
        update_interval = datetime.timedelta(days=1)

        # Init the retrieval base class
        Retrieval.__init__(
            self,
            name,
            url_csv,
            [_data_dir_fallback + "/" + name + "_fallback.csv.gz"],
            update_interval,
            **kwargs,
        )

        if auto_download:
            self.download_all_available_data()

    def download_all_available_data(self, force_local=False, force_download=False):
        """
        Attempts to download from the main url (self.url_csv) which was given on initialization.
        If this fails download from the fallbacks. It can also be specified to use the local files
        or to force the download. The download methods get inhereted from the base retrieval class.

        Parameters
        ----------
        force_local : bool, optional
            If True forces to load the local files.
        force_download : bool, optional
            If True forces the download of new files
        """
        if force_local and force_download:
            raise ValueError("force_local and force_download cant both be True!!")

        # ------------------------------------------------------------------------------ #
        # 1 Download or get local file
        # ------------------------------------------------------------------------------ #
        retrieved_local = False
        if self._timestamp_local_old(force_local) or force_download:
            self._download_helper(**self.kwargs)
        else:
            retrieved_local = self._local_helper()

        # ------------------------------------------------------------------------------ #
        # 2 Save local
        # ------------------------------------------------------------------------------ #
        self._save_to_local() if not retrieved_local else None

        # ------------------------------------------------------------------------------ #
        # 3 Convert to useable format
        # ------------------------------------------------------------------------------ #
        self._to_iso()

    def _to_iso(self):
        """
        Converts the data to a usable format i.e. converts all date string to
        datetime objects and some other column names.

        This is most of the time the first place one has to look at if something breaks!

        self.data -> self.data converted
        """
        try:
            df = self.data
            if "location" in df.columns:
                df = df.rename(columns={"location": "country"})
            # datetime columns
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            self.data = df

        except Exception as e:
            log.warning(f"There was an error formating the data! {e}")
            raise e
        return False

    def get_possible_countries(self):
        """
            Can be used to obtain all different possible countries in the dataset.

            Returns
            -------
            : pandas.DataFrame
        """
        return self.data["country"].unique()

    def get_total(self, value="tests", country=None, data_begin=None, data_end=None):
        """
            Retrieves all new cases from the Our World in Data dataset as a DataFrame with datetime index.
            Can be filtered by value, country and state, if only a country is given all available states get summed up.

            Parameters
            ----------
            value: str
                Which data to return, possible values are
                - "confirmed",
                - "tests",
                - "deaths"
                (default: "confirmed")
            country : str
                name of the country
            begin_date : datetime.datetime, optional
                intial date for the returned data, if no value is given the first date in the dataset is used
            end_date : datetime.datetime, optional
                last date for the returned data, if no value is given the most recent date in the dataset is used

            Returns
            -------
            : pandas.DataFrame
                table with new cases and the date as index
        """
        if value not in ["confirmed", "deaths", "tests"]:
            log.warning("Valid values are 'confirmed' and 'deaths'")
            raise ValueError("No valid value given! " + value)
        if value == "confirmed":
            filter_value = "total_cases"
        if value == "deaths":
            filter_value = "total_deaths"
        if value == "tests":
            filter_value = "total_tests"
        return self._filter(
            value=filter_value,
            country=country,
            data_begin=data_begin,
            data_end=data_end,
        ).dropna()

    def get_new(self, value="tests", country=None, data_begin=None, data_end=None):
        """
            Retrieves all new cases from the Our World in Data dataset as a DataFrame with datetime index.
            casesn be filtered by value, country and state, if only a country is given all available states get summed up.

            Parameters
            ----------
            value: str
                Which data to return, possible values are
                - "confirmed",
                - "tests",
                - "deaths"
                (default: "confirmed")
            country : str
                name of the country
            begin_date : datetime.datetime, optional
                intial date for the returned data, if no value is given the first date in the dataset is used
            end_date : datetime.datetime, optional
                last date for the returned data, if no value is given the most recent date in the dataset is used

            Returns
            -------
            : pandas.DataFrame
                table with new cases and the date as index
        """
        if value not in ["confirmed", "deaths", "tests"]:
            log.warning("Valid values are 'confirmed' and 'deaths'")
            raise ValueError("No valid value given! " + value)
        if value == "confirmed":
            filter_value = "new_cases"
        if value == "deaths":
            filter_value = "new_deaths"
        if value == "tests":
            filter_value = "new_tests"
        return self._filter(
            value=filter_value,
            country=country,
            data_begin=data_begin,
            data_end=data_end,
        ).dropna()

    def _filter(self, value="new_cases", country=None, data_begin=None, data_end=None):
        """
        Filter the dataset by value, country and date.
        """
        if country not in self.data["country"].unique():
            log.warning(
                "Please select a valid country. For the full dataset use self.data!"
            )
            raise ValueError("No valid country given! " + country)

        if value not in self.data.columns:
            log.warning(
                "Please select a valid filter value. For the full dataset use self.data!"
            )
            raise ValueError("No valid value given! " + value)

        # First we filter by the given country
        df = self.data.loc[self.data["country"] == country]
        # Than we get the corresponding value
        df = df[value]

        if data_begin is None:
            data_begin = df.index[0]
        if data_end is None:
            data_end = df.index[-1]

        return df[data_begin:data_end]
