import datetime
import pandas as pd
import logging

# Import base class
from .retrieval import Retrieval, _data_dir_fallback

log = logging.getLogger(__name__)


class GOOGLE(Retrieval):
    """
    This class can be used to retrieve the mobility dataset from
    `Google <https://coronavirus.jhu.edu/>`_.

    Example
    -------
    .. code-block::

        gl = cov19.data_retrieval.GOOGLE()
        gl.download_all_available_data()

        #Acess the data by
        gl.data
        #or
        gl.get_changes(filter)
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
        name = "Google"

        """
        The url to the main dataset as csv, if none if supplied the fallback routines get used
        """
        url_csv = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"

        """
        Kwargs for pandas read csv
        """
        kwargs = {"low_memory": False}  # Surpress warning

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
            if (
                "country_region" in df.columns
                and "sub_region_1" in df.columns
                and "sub_region_2" in df.columns
            ):
                df = df.rename(
                    columns={
                        "country_region": "country",
                        "sub_region_1": "state",
                        "sub_region_2": "region",
                    }
                )
            df = df.set_index(["country", "state", "region"])
            # datetime columns
            df["date"] = pd.to_datetime(df["date"])
            self.data = df
            return True
        except Exception as e:
            log.warning(f"There was an error formating the data! {e}")
            raise e
        return False

    def get_changes(
        self,
        country: str,
        state: str = None,
        region: str = None,
        data_begin: datetime.datetime = None,
        data_end: datetime.datetime = None,
    ):
        """
        Returns a dataframe with the relative changes in mobility to a baseline, provided by google.
        They are separated into "retail and recreation", "grocery and pharmacy", "parks", "transit", "workplaces" and "residental".
        Filterable for country, state and region and date.

        Parameters
        ----------
        country : str
            Selected country for the mobility data.
        state : str, optional
            State for the selected data if no value is selected the whole country is chosen
        region : str, optional
            Region for the selected data if  no value is selected the whole region/country is chosen
        data_begin, data_end : datetime.datetime, optional
            Filter for the desired time period

        Returns
        -------
        : pandas.DataFrame
        """
        if country not in self.data.index:
            raise ValueError("Invalid country!")
        if state not in self.data.index and state is not None:
            raise ValueError("Invalid state!")
        if region not in self.data.index and region is not None:
            raise ValueError("Invalid region!")
        if data_begin is not None and not isinstance(data_begin, datetime.datetime):
            raise ValueError("Invalid data_begin!")
        if data_end is not None and not isinstance(data_end, datetime.datetime):
            raise ValueError("Invalid end_date!")

        # Select everything with that country
        if state is None:
            df = self.data.iloc[self.data.index.get_level_values("region").isnull()]
        else:
            df = self.data.iloc[self.data.index.get_level_values("region") == region]

        if state is None:
            df = df.iloc[df.index.get_level_values("state").isnull()]
        else:
            df = df.iloc[df.index.get_level_values("state") == state]

        df = df.iloc[df.index.get_level_values("country") == country]

        df = df.set_index("date")

        return df.drop(columns=["country_region_code"])[data_begin:data_end]

    def get_possible_counties_states_regions(self):
        """
        Can be used to obtain all different possible countries with there corresponding possible states and regions.

        Returns
        -------
        : pandas.DataFrame
        """
        return self.data.index.unique()
