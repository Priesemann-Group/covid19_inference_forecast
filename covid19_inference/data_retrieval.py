import datetime
import os

import numpy as np
import pandas as pd


def download_data_jhu_global():
    """
        Attempts to download the most current data from the online repository of the
        Coronavirus Visual Dashboard operated by the Johns Hopkins University
        and falls back to the backup provided with our repo if it fails.

        Returns
        -------
        : confirmed_cases
            panda table with confirmed cases

        : deaths
            panda table with reported deaths
    """
    try:
        confirmed_cases_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
        deaths_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
        confirmed_cases = pd.read_csv(confirmed_cases_url, sep=",")
        deaths = pd.read_csv(deaths_url, sep=",")
    except Exception as e:
        print("Failed to download current data, using local copy.")
        this_dir = os.path.dirname(__file__)
        confirmed_cases = pd.read_csv(
            this_dir + "/data/confirmed_global_fallback_2020-03-29.csv", sep=","
        )
        deaths = pd.read_csv(
            this_dir + "/data/confirmed_global_fallback_2020-03-29.csv", sep=","
        )

    return confirmed_cases, deaths

def get_jhu_confirmed_cases():
    confirmed_cases_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    confirmed_cases = pd.read_csv(confirmed_cases_url, sep=',')
    return confirmed_cases

def get_jhu_deaths():
    deaths_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    deaths = pd.read_csv(deaths_url, sep=',')
    return deaths

def filter_one_country(data_df, country, begin_date, end_date):
    date_formatted_begin = _format_date(begin_date)
    date_formatted_end = _format_date(end_date)
    cases_obs = np.array(data_df.loc[data_df["Country/Region"] == country,
                         date_formatted_begin:date_formatted_end])[0]
    return cases_obs


def get_last_date(data_df):
    last_date = data_df.columns[-1]
    month, day, year = map(int, last_date.split('/'))
    return datetime.datetime(year + 2000, month, day)

_format_date = lambda date_py: '{}/{}/{}'.format(date_py.month, date_py.day,
                                                 str(date_py.year)[2:4])
