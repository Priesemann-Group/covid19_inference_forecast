import datetime
import os
import logging
import tempfile
import platform
import stat
import pickle

import numpy as np
import pandas as pd

import urllib, json

log = logging.getLogger(__name__)


# set by user, or default temp
_data_dir = None
# provided with the module
_data_dir_fallback = os.path.normpath(os.path.dirname(__file__) + "/../../data/")

_format_date = lambda date_py: "{}/{}/{}".format(
    date_py.month, date_py.day, str(date_py.year)[2:4]
)


def set_data_dir(fname=None, permissions=None):
    """
        Set the global variable _data_dir. New downloaded data is placed there.
        If no argument provided we try the default tmp directory.
        If permissions are not provided, uses defaults if fname is in user folder.
        If not in user folder, tries to set 777.
    """

    target = "/tmp" if platform.system() == "Darwin" else tempfile.gettempdir()

    if fname is None:
        fname = f"{target}/covid19_data"
    else:
        try:
            fname = os.path.abspath(os.path.expanduser(fname))
        except Exception as e:
            log.debug("Specified file name caused an exception, using default")
            fname = f"{target}/covid19_data"

    log.debug(f"Setting global target directory to {fname}")
    fname += "/"
    os.makedirs(fname, exist_ok=True)

    try:
        log.debug(
            f"Trying to set permissions of {fname} "
            + f"({oct(os.stat(fname)[stat.ST_MODE])[-3:]}) "
            + f"to {'defaults' if permissions is None else str(permissions)}"
        )
        dirusr = os.path.abspath(os.path.expanduser("~"))
        if permissions is None:
            if not fname.startswith(dirusr):
                os.chmod(fname, 0o777)
        else:
            os.chmod(fname, int(str(permissions), 8))
    except Exception as e:
        log.debug(f"Unable set permissions of {fname}")

    global _data_dir
    _data_dir = fname
    log.debug(f"Target directory set to {_data_dir}")
    log.debug(f"{fname} (now) has permissions {oct(os.stat(fname)[stat.ST_MODE])[-3:]}")


def get_data_dir():
    if _data_dir is None or not os.path.exists(_data_dir):
        set_data_dir()
    return _data_dir


def iso_3166_add_alternative_name_to_iso_list(
    country_in_iso_3166: str, alternative_name: str
):
    this_dir = get_data_dir()
    try:
        data = json.load(open(this_dir + "/iso_countries.json", "r"))
    except Exception as e:
        data = json.load(open(_data_dir_fallback + "/iso_countries.json", "r"))

    try:
        data[country_in_iso_3166].append(alternative_name)
        log.info("Added alternative '{alternative_name}' to {country_in_iso_3166}.")
    except Exception as e:
        raise e

    json.dump(
        data,
        open(this_dir + "/iso_countries.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4,
    )


def iso_3166_convert_to_iso(country_column_df):
    country_column_df = country_column_df.apply(
        lambda x: x
        if iso_3166_country_in_iso_format(x)
        else iso_3166_get_country_name_from_alternative(x)
    )
    return country_column_df


def iso_3166_get_country_name_from_alternative(alternative_name: str) -> str:
    this_dir = get_data_dir()
    try:
        data = json.load(open(this_dir + "/iso_countries.json", "r"))
    except Exception as e:
        data = json.load(open(_data_dir_fallback + "/iso_countries.json", "r"))

    for country, alternatives in data.items():
        for alt in alternatives:
            if alt == alternative_name:
                return country
    log.debug(
        f"Alternative_name '{str(alternative_name)}' not found in iso convertion list!"
    )
    return alternative_name


def iso_3166_country_in_iso_format(country: str) -> bool:
    this_dir = get_data_dir()
    try:
        data = json.load(open(this_dir + "/iso_countries.json", "r"))
    except Exception as e:
        data = json.load(open(_data_dir_fallback + "/iso_countries.json", "r"))
    if country in data:
        return True
    return False


def backup_instances(
    trace=None, model=None, fname="latest_",
):
    """
        helper to save or load trace and model instances.
        loads from `fname` if provided traces and model variables are None,
        else saves them there.
    """

    try:
        if trace is None and model is None:
            with open(f"{get_data_dir()}{fname}_model.pickle", "rb") as handle:
                model = pickle.load(handle)
            with open(f"{get_data_dir()}{fname}_trace.pickle", "rb") as handle:
                trace = pickle.load(handle)
        else:
            with open(f"{get_data_dir()}{fname}_model.pickle", "wb") as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f"{get_data_dir()}{fname}_trace.pickle", "wb") as handle:
                pickle.dump(trace, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        log.info(f"Failed to backup instances of model and trace: {e}")
        trace = None
        model = None

    return model, trace


class Retrieval:
    """
        Each source class should inherit this base retrieval class, it streamlines alot
        of base functions. It manages downloads, multiple fallbacks and local backups
        via timestamp. At init of the parent class the Retrieval init should be called
        with the following arguments, these get saved as attributes.

        An example for the usage can be seen in the _Google, _RKI and _JHU source files.
    """

    url_csv = ""

    fallbacks = []

    name = ""

    update_interval = datetime.timedelta(days=1)

    def __init__(self, name, url_csv, fallbacks, update_interval=None, **kwargs):
        """
        Parameters
        ----------
        name : str
            A name for the Parent class, mainly used for the local file backup.
        url_csv : str
            The url to the main dataset as csv, if an empty string if supplied the fallback routines get used.
        fallbacks : array
            Fallbacks can be filepaths to local or online sources
            or even methods defined in the parent class.
        update_interval : datetime.timedelta
            If the local file is older than the update_interval it gets updated once the
            download all function is called.
        """
        self.name = name
        self.url_csv = url_csv
        self.fallbacks = fallbacks
        self.kwargs = kwargs

        if update_interval is not None:
            self.update_interval = update_interval

    def _download_csv_from_source(self, filepath, **kwargs):
        """
        Uses pandas read csv to download the csv file.
        The possible kwargs can be seen in the pandas `documentation <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html#pandas.read_csv>`_.

        These kwargs can vary for the different parent classes and should be defined there!

        Parameter
        ---------
        filepath : str
            Full path to the desired csv file

        Return
        ------
        :bool
            True if the retrieval was a success, False if it failed
        """
        self.data = pd.read_csv(filepath, **kwargs)
        return True

    def _fallback_handler(self):
        """
        Recursivly iterate over all fallbacks and try to execute subroutines depending on the
        type of fallback.
        """

        def execute_fallback(fallback, i):
            """Helper function to execute the subroutines depending on the type"""
            # Break condition
            success = False
            try:
                # Try to execute the fallback
                if callable(fallback):
                    success = fallback()
                # If it is not executable we try to download from the source
                elif isinstance(fallback, str):
                    success = self._download_csv_from_source(fallback, **self.kwargs)
                else:
                    log.info(
                        f"That is weird fallback is not of type string nor a callable function {type(fallback)}"
                    )
                    raise Exception(
                        f"Fallback type not supported (yet?) {type(fallback)}"
                    )
            except Exception as e:
                log.info(f"Fallback {i} failed! {fallback}:{e}")

            # ---------------------------------------------------------------#
            # Break conditions
            # ---------------------------------------------------------------#
            if success:
                log.debug(f"Fallback {i} successful! {fallback}")
                return True
            if len(self.fallbacks) == i + 1:
                log.warning(f"ALL fallbacks failed! This should not happen!")
                return False

            # ---------------------------------------------------------------#
            # Continue Recursion
            # ---------------------------------------------------------------#
            execute_fallback(self.fallbacks[i + 1], i + 1)

        # Start Recursion
        success = execute_fallback(self.fallbacks[0], 0)
        return success

    def _timestamp_local_old(self, force_local=False) -> bool:
        """
        1. Get timestamp if it exists
        2. compare with the date today
        3. update if data is older than set intervall -> can be parent dependant
        """
        if not os.path.isfile(get_data_dir() + self.name + "_timestamp.json"):
            return True

        if force_local:
            return False

        timestamp = json.load(open(get_data_dir() + self.name + "_timestamp.json", "r"))
        timestamp = datetime.datetime.strptime(timestamp, "%m/%d/%Y, %H:%M:%S")

        if (datetime.datetime.now() - timestamp) > self.update_interval:
            log.debug("Timestamp old. Trying to download new files")
            return True

        return False

    def _download_helper(self, **kwargs):
        # First we check if the date of the online file is newer and if we have to download a new file
        # this is done by a function which can be seen above
        try:
            # Try to download from original souce
            self._download_csv_from_source(self.url_csv, **kwargs)
        except Exception as e:
            # Try all fallbacks
            log.info(f"Failed to download from url {self.url_csv} : {e}")
            self._fallback_handler()
        finally:
            # We save it to the local files
            # self.data._save_to_local()
            log.info(f"Successfully downloaded new files.")

    def _local_helper(self):
        # If we can use a local file we construct the path from the given local name
        try:
            self._download_csv_from_source(
                get_data_dir() + self.name + ".csv.gz", **self.kwargs
            )
            log.info(f"Successfully loaded data from local")
            return True
        except Exception as e:
            log.info(f"Failed to load local files! {e} Trying fallbacks!")
            self.download_helper(**self.kwargs)
        return False

    def _save_to_local(self):
        """
        Creates a local backup for the self.data pandas.DataFrame. And a timestamp for the source.
        """

        filepath = get_data_dir() + self.name + ".csv.gz"
        try:
            self.data.to_csv(filepath, compression="infer", index=False)
            self._create_timestamp()
            log.info(f"Local backup to {filepath} successful.")
            return True
        except Exception as e:
            log.warning(f"Could not create local backup {e}")
            raise e
        return False

    def _create_timestamp(self):
        try:
            timestamp = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            json.dump(
                timestamp,
                open(
                    get_data_dir() + self.name + "_timestamp.json",
                    "w",
                    encoding="utf-8",
                ),
                ensure_ascii=False,
                indent=4,
            )
        except Exception as e:
            raise e
