import datetime
import pandas as pd
import logging

# Import base class
from .retrieval import Retrieval

log = logging.getLogger(__name__)


class RKIsituationreports(Retrieval):
    """
    As mentioned by Matthias Linden, the daily situation reports have more available data.
    This class retrieves this additional data from Matthias website and parses it into the format we use i.e. a datetime index.

    Interesting new data is for example ICU cases, deaths and recorded symptoms. For now one can look at the data by running

    Example
    -------
    .. code-block::

        rki_si_re = cov19.data_retrieval.RKIsituationreports(True)
        print(rki_si_re.data)

    ToDo
    -----
    Filter functions for ICU, Symptoms and maybe even daily new cases for the respective categories.

    """

    def __init__(self, auto_download=False):
        """
        On init of this class the base Retrieval Class __init__ is called, with rki situation reports specific
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
        name = "Rki_sit_rep"

        """
        The url to the main dataset as csv, if none if supplied the fallback routines get used
        """
        url_csv = "http://mlinden.de/COVID19/data/latest_report.csv"

        """
        Kwargs for pandas read csv
        """
        kwargs = {"sep": ";"}

        """
        If the local file is older than the update_interval it gets updated once the
        download all function is called. Can be diffent values depending on the parent class
        """
        update_interval = datetime.timedelta(days=1)

        # Init the retrieval base class
        Retrieval.__init__(self, name, url_csv, [], update_interval, **kwargs)

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
        try:
            df = self.data
            if "Unnamed: 0" in df.columns:
                df["date"] = pd.to_datetime(df["Unnamed: 0"])
                df = df.drop(columns="Unnamed: 0")
            df = df.set_index(["date"])
            self.data = df
            return True
        except Exception as e:
            log.warning(f"There was an error formating the data! {e}")
            raise e
        return False
