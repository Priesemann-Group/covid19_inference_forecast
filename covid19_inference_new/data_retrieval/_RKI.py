import pandas as pd
import datetime
import logging
import numpy as np

# Import base class
from .retrieval import Retrieval, get_data_dir, _data_dir_fallback

import urllib, json


log = logging.getLogger(__name__)


class RKI(Retrieval):
    """
    This class can be used to retrieve and filter the dataset from the Robert Koch Institute `Robert Koch Institute <https://www.rki.de/>`_.
    The data gets retrieved from the `arcgis <https://www.arcgis.com/sharing/rest/content/items/f10774f1c63e40168479a1feb6c7ca74/data>`_  dashboard.

    Features
        - download the full dataset
        - filter by date
        - filter by bundesland
        - filter by recovered, deaths and confirmed cases

    Example
    -------
    .. code-block::

        rki = cov19.data_retrieval.RKI()
        rki.download_all_available_data()

        #Acess the data by
        rki.data
        #or
        rki.get_new("confirmed","Sachsen")
        rki.get_total(filter)
    """

    def __init__(self, auto_download=False):
        """
        On init of this class the base Retrieval Class __init__ is called, with rki specific
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
        name = "Rki"

        """
        The url to the main dataset as csv, if none if supplied the fallback routines get used
        """
        url_csv = "https://www.arcgis.com/sharing/rest/content/items/f10774f1c63e40168479a1feb6c7ca74/data"

        """
        Kwargs for pandas read csv
        """
        kwargs = {}  # Surpress warning

        """
        fallback array can be anything a filepath or callable methods
        """
        fallbacks = [
            self.__download_via_rest_api,
            _data_dir_fallback + "/" + name + "_fallback.csv.gz",
        ]
        """
        If the local file is older than the update_interval it gets updated once the
        download all function is called. Can be diffent values depending on the parent class
        """
        update_interval = datetime.timedelta(days=1)

        # Init the retrieval base class
        Retrieval.__init__(self, name, url_csv, fallbacks, update_interval, **kwargs)

        self.data = None

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
        df = self.data
        if "Meldedatum" in df.columns:
            df["date"] = df["Meldedatum"].apply(
                lambda x: datetime.datetime.strptime(x, "%Y/%m/%d %H:%M:%S")
            )
            df = df.drop(columns="Meldedatum")
        if "Refdatum" in df.columns:
            df["date_ref"] = df["Refdatum"].apply(
                lambda x: datetime.datetime.strptime(x, "%Y/%m/%d %H:%M:%S")
            )
            df = df.drop(columns="Refdatum")

        # Rename the columns to match the JHU dataset
        if "AnzahlFall" in df.columns:
            df.rename(columns={"AnzahlFall": "confirmed"}, inplace=True)
        if "AnzahlTodesfall" in df.columns:
            df.rename(columns={"AnzahlTodesfall": "deaths"}, inplace=True)
        if "AnzahlGenesen" in df.columns:
            df.rename(columns={"AnzahlGenesen": "recovered"}, inplace=True)

        df["date"] = pd.to_datetime(df["date"])
        df["date_ref"] = pd.to_datetime(df["date_ref"])
        self.data = df

    def __download_via_rest_api(self, try_max=10):
        landkreise_max = 412  # Strangely there are 412 regions defined by the Robert Koch Insitute in contrast to the offical 294 rural districts or the 401 administrative districts.
        url_id = "https://services7.arcgis.com/mOBPykOjAyBO2ZKk/ArcGIS/rest/services/RKI_COVID19/FeatureServer/0/query?where=0%3D0&objectIds=&time=&resultType=none&outFields=idLandkreis&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=true&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token="

        url = urllib.request.urlopen(url_id)
        json_data = json.loads(url.read().decode())
        n_data = len(json_data["features"])
        unique_ids = [
            json_data["features"][i]["attributes"]["IdLandkreis"] for i in range(n_data)
        ]

        # If the number of landkreise is smaller than landkreise_max, uses local copy (query system can behave weirdly during updates)
        if n_data >= landkreise_max:
            log.info(f"Downloading {n_data} unique Landkreise. May take a while.\n")
            df_keys = [
                "IdBundesland",
                "Bundesland",
                "Landkreis",
                "Altersgruppe",
                "Geschlecht",
                "AnzahlFall",
                "AnzahlTodesfall",
                "ObjectId",
                "IdLandkreis",
                "Datenstand",
                "NeuerFall",
                "NeuerTodesfall",
                "NeuGenesen",
                "AnzahlGenesen",
                "date",
                "date_ref",
            ]

            df = pd.DataFrame(columns=df_keys)

            # Fills DF with data from all landkreise
            for idlandkreis in unique_ids:

                url_str = (
                    "https://services7.arcgis.com/mOBPykOjAyBO2ZKk/ArcGIS/rest/services/RKI_COVID19/FeatureServer/0//query?where=IdLandkreis%3D"
                    + idlandkreis
                    + "&objectIds=&time=&resultType=none&outFields=Bundesland%2C+Landkreis%2C+IdBundesland%2C+ObjectId%2C+IdLandkreis%2C+Altersgruppe%2C+Geschlecht%2C+AnzahlFall%2C+AnzahlTodesfall%2C+Meldedatum%2C+NeuerFall%2C+Refdatum%2C+Datenstand%2C+NeuGenesen%2C+AnzahlGenesen&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token="
                )

                count_try = 0

                while count_try < try_max:
                    try:
                        with urllib.request.urlopen(url_str) as url:
                            json_data = json.loads(url.read().decode())

                        n_data = len(json_data["features"])

                        if n_data > 5000:
                            raise ValueError("Query limit exceeded")

                        data_flat = [
                            json_data["features"][i]["attributes"]
                            for i in range(n_data)
                        ]

                        break

                    except:
                        count_try += 1

                if count_try == try_max:
                    raise ValueError("Maximum limit of tries exceeded.")

                df_temp = pd.DataFrame(data_flat)

                # Very inneficient, but it will do
                df = pd.concat([df, df_temp], ignore_index=True)

            df["date"] = df["Meldedatum"].apply(
                lambda x: datetime.datetime.fromtimestamp(x / 1e3)
            )
            df["date_ref"] = df["Refdatum"].apply(
                lambda x: datetime.datetime.fromtimestamp(x / 1e3)
            )
            df = df.drop(columns="Meldedatum")
            df = df.drop(columns="Refdatum")

        else:
            raise RuntimeError("Invalid response from REST api")

        return df

    def get_total(
        self,
        value="confirmed",
        bundesland: str = None,
        landkreis: str = None,
        data_begin: datetime.datetime = None,
        data_end: datetime.datetime = None,
        date_type: str = "date",
    ):
        """
        Gets all total confirmed cases for a region as dataframe with date index. Can be filtered with multiple arguments.

        Parameters
        ----------
        value: str
            Which data to return, possible values are
            - "confirmed",
            - "recovered",
            - "deaths"
            (default: "confirmed")
        bundesland : str, optional
            if no value is provided it will use the full summed up dataset for Germany
        landkreis : str, optional
            if no value is provided it will use the full summed up dataset for the region (bundesland)
        data_begin : datetime.datetime, optional
            initial date, if no value is provided it will use the first possible date
        data_end : datetime.datetime, optional
            last date, if no value is provided it will use the most recent possible date
        date_type : str, optional
            type of date to use: reported date 'date' (Meldedatum in the original dataset), or symptom date 'date_ref' (Refdatum in the original dataset)

        Returns
        -------
        :pandas.DataFrame
        """

        # ------------------------------------------------------------------------------ #
        # Default parameters
        # ------------------------------------------------------------------------------ #
        if value not in ["confirmed", "recovered", "deaths"]:
            raise ValueError(
                'Invalid value. Valid options: "confirmed", "deaths", "recovered"'
            )

        if self.data is None:
            self.download_all_available_data()

        # Note: It should be fine to NOT check for the date since this is also done by the filter_date method

        # Set level for filter use bundesland if no landkreis is supplied else use landkreis
        level = None
        filter_value = None
        if bundesland is not None and landkreis is None:
            level = "Bundesland"
            filter_value = bundesland
        elif bundesland is None and landkreis is not None:
            level = "Landkreis"
            filter_value = landkreis
        elif bundesland is not None and landkreis is not None:
            raise ValueError("bundesland and landkreis cannot be simultaneously set.")

        # ------------------------------------------------------------------------------ #
        # Retrieve data and filter it
        # ------------------------------------------------------------------------------ #
        df = self.filter(data_begin, data_end, value, date_type, level, filter_value)
        return df

    def get_new(
        self,
        value="confirmed",
        bundesland: str = None,
        landkreis: str = None,
        data_begin: datetime.datetime = None,
        data_end: datetime.datetime = None,
        date_type: str = "date",
    ):
        """
        Retrieves all new cases from the Robert Koch Institute dataset as a DataFrame with datetime index.
        Can be filtered by value, bundesland and landkreis, if only a country is given all available states get summed up.

        Parameters
        ----------
        value: str
            Which data to return, possible values are
            - "confirmed",
            - "recovered",
            - "deaths"
            (default: "confirmed")
        bundesland : str, optional
            if no value is provided it will use the full summed up dataset for Germany
        landkreis : str, optional
            if no value is provided it will use the full summed up dataset for the region (bundesland)
        data_begin : datetime.datetime, optional
            intial date for the returned data, if no value is given the first date in the dataset is used,
            if none is given could yield errors
        data_end : datetime.datetime, optional
            last date for the returned data, if no value is given the most recent date in the dataset is used

        Returns
        -------
        : pandas.DataFrame
            table with daily new confirmed and the date as index
        """

        # ------------------------------------------------------------------------------ #
        # Default parameters
        # ------------------------------------------------------------------------------ #

        if value not in ["confirmed", "recovered", "deaths"]:
            raise ValueError(
                'Invalid value. Valid options: "confirmed", "deaths", "recovered"'
            )

        if self.data is None:
            self.download_all_available_data()

        level = None
        filter_value = None
        if bundesland is not None and landkreis is None:
            level = "Bundesland"
            filter_value = bundesland
        elif bundesland is None and landkreis is not None:
            level = "Landkreis"
            filter_value = landkreis
        elif bundesland is not None and landkreis is not None:
            raise ValueError("bundesland and landkreis cannot be simultaneously set.")

        if data_begin is None:
            data_begin = self.data[date_type].max()
        if data_end is None:
            data_end = self.data[date_type].min()

        if data_begin == self.data[date_type].max():
            raise ValueError(
                "Date has to be after the first dataset entry. Set a data_begin date!"
            )

        # ------------------------------------------------------------------------------ #
        # Retrieve data and filter it
        # ------------------------------------------------------------------------------ #

        df = self.filter(
            data_begin - datetime.timedelta(days=1),
            data_end,
            value,
            date_type,
            level,
            filter_value,
        )
        # Get difference to the days beforehand
        df = (
            df.diff().drop(df.index[0]).astype(int)
        )  # Neat oneliner to also drop the first row and set the type back to int
        return df.fillna(0)

    def filter(
        self,
        data_begin: datetime.datetime = None,
        data_end: datetime.datetime = None,
        variable="confirmed",
        date_type="date",
        level=None,
        value=None,
    ):
        """
        Filters the obtained dataset for a given time period and returns an array ONLY containing only the desired variable.

        Parameters
        ----------
        data_begin : datetime.datetime, optional
            initial date, if no value is provided it will use the first possible date
        data_end : datetime.datetime, optional
            last date, if no value is provided it will use the most recent possible date
        variable : str, optional
            type of variable to return
            possible types are:
            "confirmed"      : cases (default)
            "AnzahlTodesfall" : deaths
            "AnzahlGenesen"   : recovered
        date_type : str, optional
            type of date to use: reported date 'date' (Meldedatum in the original dataset), or symptom date 'date_ref' (Refdatum in the original dataset)
        level : str, optional
            possible strings are:
                "None"       : return data from all Germany (default)
                "Bundesland" : a state
                "Landkreis"  : a region
        value : None, optional
            string of the state/region
            e.g. "Sachsen"

        Returns
        -------
        : pd.DataFrame
            array with ONLY the requested variable, in the requested range. (one dimensional)
        """
        # Input parsing
        if variable not in ["confirmed", "deaths", "recovered"]:
            raise ValueError(
                'Invalid variable. Valid options: "confirmed", "deaths", "recovered"'
            )

        if level not in ["Landkreis", "Bundesland", None]:
            raise ValueError(
                'Invalid level. Valid options: "Landkreis", "Bundesland", None'
            )

        if date_type not in ["date", "date_ref"]:
            raise ValueError('Invalid date_type. Valid options: "date", "date_ref"')

        df = self.data.sort_values(date_type)
        if data_begin is None:
            data_begin = df[date_type].iloc[0]
        if data_end is None:
            data_end = df[date_type].iloc[-1]
        if not isinstance(data_begin, datetime.datetime) and isinstance(
            data_end, datetime.datetime
        ):
            raise ValueError(
                "Invalid data_begin, data_end: has to be datetime.datetime object"
            )

        # Keeps only the relevant data
        df = self.data

        if level is not None:
            df = df[df[level] == value][[date_type, variable]]

        df_series = df.groupby(date_type)[variable].sum().cumsum()
        df_series.index = pd.to_datetime(df_series.index)

        return df_series[data_begin:data_end].fillna(0)

    def filter_all_bundesland(
        self,
        begin_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
        variable="confirmed",
        date_type="date",
    ):
        """
        Filters the full RKI dataset

        Parameters
        ----------
        df : DataFrame
            RKI dataframe, from get_rki()
        begin_date : datetime.datetime
            initial date to return
        end_date : datetime.datetime
            last date to return
        variable : str, optional
            type of variable to return: cases ("AnzahlFall"), deaths ("AnzahlTodesfall"), recovered ("AnzahlGenesen")
        date_type : str, optional
            type of date to use: reported date 'date' (Meldedatum in the original dataset), or symptom date 'date_ref' (Refdatum in the original dataset)

        Returns
        -------
        : pd.DataFrame
            DataFrame with datetime dates as index, and all German regions (bundesländer) as columns
        """
        if variable not in ["confirmed", "deaths", "recovered"]:
            raise ValueError(
                'Invalid variable. Valid options: "confirmed", "deaths", "recovered"'
            )

        if date_type not in ["date", "date_ref"]:
            raise ValueError('Invalid date_type. Valid options: "date", "date_ref"')

        if begin_date is None:
            begin_date = self.data[date_type].iloc[0]
        if end_date is None:
            end_date = self.data[date_type].iloc[-1]

        if not isinstance(begin_date, datetime.datetime) and isinstance(
            end_date, datetime.datetime
        ):
            raise ValueError(
                "Invalid begin_date, end_date: has to be datetime.datetime object"
            )

        # Nifty, if slightly unreadable one-liner
        df = self.data
        df2 = (
            df.groupby([date_type, "Bundesland"])[variable]
            .sum()
            .reset_index()
            .pivot(index=date_type, columns="Bundesland", values=variable)
            .fillna(0)
        )
        df2.index = pd.to_datetime(df2.index)
        # Returns cumsum of variable
        return df2[begin_date:end_date].cumsum()
