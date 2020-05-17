import datetime
import pandas as pd
import logging

# Import base class
from .retrieval import Retrieval, get_data_dir, _data_dir_fallback

log = logging.getLogger(__name__)


class JHU(Retrieval):
    """
    This class can be used to retrieve and filter the dataset from the online repository of the coronavirus visual dashboard operated
    by the `Johns Hopkins University <https://coronavirus.jhu.edu/>`_.

    Features
        - download all files from the online repository of the coronavirus visual dashboard operated by the Johns Hopkins University.
        - filter by deaths, confirmed cases and recovered cases
        - filter by country and state
        - filter by date

    Example
    -------
    .. code-block::

        jhu = cov19.data_retrieval.JHU()
        jhu.download_all_available_data()

        #Acess the data by
        jhu.data
        #or
        jhu.get_new("confirmed","Italy")
        jhu.get_total(filter)
    """

    @property
    def data(self):
        if self.confirmed is None or self.deaths is None or self.recovered is None:
            return None
        return (self.confirmed, self.deaths, self.recovered)

    def __init__(self, auto_download=False):
        """
        On init of this class the base Retrieval Class __init__ is called, with jhu specific
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
        name = "Jhu"

        """
        The url to the main dataset as csv, if none if supplied the fallback routines get used
        """
        url_csv = [
            "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",
            "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv",
            "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv",
        ]

        """
        Kwargs for pandas read csv
        """
        kwargs = {}  # Surpress warning

        """
        Fallbacks
        """
        fallbacks = [self._fallback_local_backup]

        """
        If the local file is older than the update_interval it gets updated once the
        download all function is called. Can be diffent values depending on the parent class
        """
        update_interval = datetime.timedelta(days=1)

        # Init the retrieval base class
        Retrieval.__init__(self, name, url_csv, fallbacks, update_interval, **kwargs)

        self.confirmed = None
        self.deaths = None
        self.recovered = None

        if auto_download:
            self.download_all_available_data()

    def download_all_available_data(self, force_local=False, force_download=False):
        """
        Attempts to download from the main urls (self.url_csv) which was set on initialization of
        this class.
        If this fails it downloads from the fallbacks. It can also be specified to use the local files
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

        def helper(df):
            try:
                df = df.drop(columns=["Lat", "Long"]).rename(
                    columns={"Province/State": "state", "Country/Region": "country"}
                )
                df = df.set_index(["country", "state"])
                df.columns = pd.to_datetime(df.columns)
            except Exception as e:
                log.warning(f"There was an error formating the data! {e}")
                raise e
            return df

        self.confirmed = helper(self.confirmed).T
        self.deaths = helper(self.deaths).T
        self.recovered = helper(self.recovered).T

        return True

    def get_total_confirmed_deaths_recovered(
        self,
        country: str = None,
        state: str = None,
        begin_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
    ):
        """
        Retrieves all confirmed, deaths and recovered cases from the Johns Hopkins University dataset as a DataFrame with datetime index.
        Can be filtered by country and state, if only a country is given all available states get summed up.

        Parameters
        ----------
        country : str, optional
            name of the country (the "Country/Region" column), can be None if the whole summed up data is wanted (why would you do this?)
        state : str, optional
            name of the state (the "Province/State" column), can be None if country is set or the whole summed up data is wanted
        begin_date : datetime.datetime, optional
            intial date for the returned data, if no value is given the first date in the dataset is used
        end_date : datetime.datetime, optional
            last date for the returned data, if no value is given the most recent date in the dataset is used

        Returns
        -------
        : pandas.DataFrame
        """

        # filter
        df = pd.DataFrame(
            columns=["date", "confirmed", "deaths", "recovered"]
        ).set_index("date")
        if country is None:
            df["confirmed"] = self.confirmed.sum(axis=1, skipna=True)
            df["deaths"] = self.deaths.sum(axis=1, skipna=True)
            df["recovered"] = self.recovered.sum(axis=1, skipna=True)
        else:
            if state is None:
                df["confirmed"] = self.confirmed[country].sum(axis=1, skipna=True)
                df["deaths"] = self.deaths[country].sum(axis=1, skipna=True)
                df["recovered"] = self.recovered[country].sum(axis=1, skipna=True)
            else:
                df["confirmed"] = self.confirmed[(country, state)]
                df["deaths"] = self.deaths[(country, state)]
                df["recovered"] = self.recovered[(country, state)]
        df.index.name = "date"

        return self.filter_date(df, begin_date, end_date)

    def get_new(
        self,
        value="confirmed",
        country: str = None,
        state: str = None,
        data_begin: datetime.datetime = None,
        data_end: datetime.datetime = None,
    ):
        """
        Retrieves all new cases from the Johns Hopkins University dataset as a DataFrame with datetime index.
        Can be filtered by value, country and state, if only a country is given all available states get summed up.

        Parameters
        ----------
        value: str
            Which data to return, possible values are
            - "confirmed",
            - "recovered",
            - "deaths"
            (default: "confirmed")
        country : str, optional
            name of the country (the "Country/Region" column), can be None
        state : str, optional
            name of the state (the "Province/State" column), can be None
        begin_date : datetime.datetime, optional
            intial date for the returned data, if no value is given the first date in the dataset is used
        end_date : datetime.datetime, optional
            last date for the returned data, if no value is given the most recent date in the dataset is used

        Returns
        -------
        : pandas.DataFrame
            table with new cases and the date as index

        """

        # ------------------------------------------------------------------------------ #
        # Default Parameters
        # ------------------------------------------------------------------------------ #
        if value not in ["confirmed", "recovered", "deaths"]:
            raise ValueError(
                'Invalid value. Valid options: "confirmed", "deaths", "recovered"'
            )

        if self.data is None:
            self.download_all_available_data()

        if country == "None":
            country = None
        if state == "None":
            state = None

        # If no date is given set to first and last dates in data
        if data_begin is None:
            data_begin = self.__get_first_date()
        if data_end is None:
            data_end = self.__get_last_date()

        if data_begin == self.data[0].index[0]:
            raise ValueError("Date has to be after the first dataset entry")

        # ------------------------------------------------------------------------------ #
        # Retrieve data and filter it
        # ------------------------------------------------------------------------------ #
        df = pd.DataFrame(columns=["date", value]).set_index("date")

        if country is None:
            df[value] = orig.sum(axis=1, skipna=True)
        else:
            if state is None:
                df[value] = getattr(self, value)[country].sum(axis=1, skipna=True)
            else:
                df[value] = getattr(self, value)[(country, state)]
        df.index.name = "date"

        df = self.filter_date(df, data_begin - datetime.timedelta(days=1), data_end)
        df = (
            df.diff().drop(df.index[0]).astype(int)
        )  # Neat oneliner to also drop the first row and set the type back to int
        return df[value]

    def get_total(
        self,
        value="confirmed",
        country: str = None,
        state: str = None,
        data_begin: datetime.datetime = None,
        data_end: datetime.datetime = None,
    ):
        """
        Retrieves all total/cumulative cases from the Johns Hopkins University dataset as a DataFrame with datetime index.
        Can be filtered by value, country and state, if only a country is given all available states get summed up.

        Parameters
        ----------
        value: str
            Which data to return, possible values are
            - "confirmed",
            - "recovered",
            - "deaths"
            (default: "confirmed")
        country : str, optional
            name of the country (the "Country/Region" column), can be None
        state : str, optional
            name of the state (the "Province/State" column), can be None
        begin_date : datetime.datetime, optional
            intial date for the returned data, if no value is given the first date in the dataset is used
        end_date : datetime.datetime, optional
            last date for the returned data, if no value is given the most recent date in the dataset is used

        Returns
        -------
        : pandas.DataFrame
            table with total/cumulative cases and the date as index
        """

        # ------------------------------------------------------------------------------ #
        # Default Parameters
        # ------------------------------------------------------------------------------ #
        if value not in ["confirmed", "recovered", "deaths"]:
            raise ValueError(
                'Invalid value. Valid options: "confirmed", "deaths", "recovered"'
            )

        if self.data is None:
            self.download_all_available_data()

        if country == "None":
            country = None
        if state == "None":
            state = None

        # Note: It should be fine to NOT check for the date since this is also done by the filter_date method

        # ------------------------------------------------------------------------------ #
        # Retrieve data and filter it
        # ------------------------------------------------------------------------------ #
        df = pd.DataFrame(columns=["date", value]).set_index("date")
        orig = getattr(self, value)
        if country is None:
            df[value] = getattr(self, value).sum(axis=1, skipna=True)
        else:
            if state is None:
                df[value] = getattr(self, value)[country].sum(axis=1, skipna=True)
            else:
                df[value] = getattr(self, value)[(country, state)]
        df.index.name = "date"
        df = self.filter_date(df, data_begin, data_end)
        return df[value]

    def filter_date(
        self,
        df,
        begin_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
    ):
        """
        Returns give dataframe between begin and end date. Dataframe has to have a datetime index.

        Parameters
        ----------
        begin_date : datetime.datetime, optional
            First day that should be filtered
        end_date : datetime.datetime, optional
            Last day that should be filtered

        Returns
        -------
        : pandas.DataFrame
        """
        if begin_date is None:
            begin_date = self.__get_first_date()
        if end_date is None:
            end_date = self.__get_last_date()

        if not isinstance(begin_date, datetime.datetime) and isinstance(
            end_date, datetime.datetime
        ):
            raise ValueError(
                "Invalid begin_date, end_date: has to be datetime.datetime object"
            )

        return df[begin_date:end_date]

    def __get_first_date(self):
        return self.data[0].index[0]

    def __get_last_date(self):
        return self.data[0].index[-1]

    def get_possible_countries_states(self):
        """
        Can be used to get a list with all possible states and coutries.

        Returns
        -------
        : pandas.DataFrame in the format
        """
        all_entrys = (
            list(self.confirmed.columns)
            + list(self.deaths.columns)
            + list(self.recovered.columns)
        )
        df = pd.DataFrame(all_entrys, columns=["country", "state"])

        return df

    # ------------------------------------------------------------------------------ #
    # Helper methods, overload from the base class
    # ------------------------------------------------------------------------------ #
    def _download_helper(self, **kwargs):
        """
        Overloads the method method from the Base Retrival class
        """
        try:
            # Try to download from original souce
            self._download_csvs_from_source(self.url_csv, **kwargs)
        except Exception as e:
            # Try all fallbacks
            log.info(f"Failed to download from url {self.url_csv} : {e}")
            self._fallback_handler()
        finally:
            # We save it to the local files
            # self.data._save_to_local()
            log.info(f"Successfully downloaded new files.")

    def _local_helper(self):
        """
        Overloads the method method from the Base Retrival class
        """
        try:
            self._download_csvs_from_source(
                [
                    get_data_dir() + self.name + "_confirmed" + ".csv.gz",
                    get_data_dir() + self.name + "_deaths" + ".csv.gz",
                    get_data_dir() + self.name + "_recovered" + ".csv.gz",
                ],
                **self.kwargs,
            )
            log.info(f"Successfully loaded data from local")
            return True
        except Exception as e:
            log.info(f"Failed to load local files! {e} Trying fallbacks!")
            self.download_helper(**self.kwargs)
        return False

    def _save_to_local(self):
        """
        Overloads the method method from the Base Retrival class
        """
        filepaths = [
            get_data_dir() + self.name + "_confirmed" + ".csv.gz",
            get_data_dir() + self.name + "_deaths" + ".csv.gz",
            get_data_dir() + self.name + "_recovered" + ".csv.gz",
        ]
        try:
            self.confirmed.to_csv(filepaths[0], compression="infer", index=False)
            self.deaths.to_csv(filepaths[1], compression="infer", index=False)
            self.recovered.to_csv(filepaths[2], compression="infer", index=False)
            self._create_timestamp()
            log.info(f"Local backup to {filepaths} successful.")
            return True
        except Exception as e:
            log.warning(f"Could not create local backup {e}")
            raise e
        return False

    def _download_csvs_from_source(self, filepaths, **kwargs):
        self.confirmed = pd.read_csv(filepaths[0], **kwargs)
        self.deaths = pd.read_csv(filepaths[1], **kwargs)
        self.recovered = pd.read_csv(filepaths[2], **kwargs)

    def _fallback_local_backup(self):
        path_confirmed = (
            _data_dir_fallback
            + "/"
            + self.name
            + "_confirmed"
            + "_fallback"
            + ".csv.gz"
        )
        path_deaths = (
            _data_dir_fallback + "/" + self.name + "_deaths" + "_fallback" + ".csv.gz"
        )
        path_recovered = (
            _data_dir_fallback
            + "/"
            + self.name
            + "_recovered"
            + "_fallback"
            + ".csv.gz"
        )
        self.confirmed = pd.read_csv(path_confirmed, **self.kwargs)
        self.deaths = pd.read_csv(path_deaths, **self.kwargs)
        self.recovered = pd.read_csv(path_recovered, **self.kwargs)
