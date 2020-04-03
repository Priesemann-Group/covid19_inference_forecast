import datetime

import numpy as np
import pandas as pd

def get_jhu_confirmed_cases():
    confirmed_cases_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    confirmed_cases = pd.read_csv(confirmed_cases_url, sep=',')
    return confirmed_cases

def get_jhu_deaths():
    deaths_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    deaths = pd.read_csv(deaths_url, sep=',')
    return deaths

def filter_one_country(data_df, country, begin_date, end_date):
    date_formatted_begin = format_date(begin_date)
    date_formatted_end = format_date(end_date)
    cases_obs = np.array(data_df.loc[data_df["Country/Region"] == country,
                         date_formatted_begin:date_formatted_end])[0]
    return cases_obs


def get_last_date(data_df):
    last_date = data_df.columns[-1]
    month, day, year = map(int, last_date.split('/'))
    return datetime.datetime(year + 2000, month, day)

format_date = lambda date_py: '{}/{}/{}'.format(date_py.month, date_py.day,
                                                 str(date_py.year)[2:4])