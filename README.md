# Covid-19 inference and forecast

[![Documentation Status](https://readthedocs.org/projects/covid19-inference-forecast/badge/?version=latest)](https://covid19-inference-forecast.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

We attempt to model the spread of Covid-19.


The current paper draft (as of 2020-04-01) can be downloaded [from the repo.](https://github.com/Priesemann-Group/covid19_inference_forecast/raw/master/paper_draft_2020-04-01.pdf)

The current research article (a short version) [is also available.](https://github.com/Priesemann-Group/covid19_inference_forecast/raw/master/paper_overview_2020-04-01.pdf)

The code used to produce the figures is available [here](https://github.com/Priesemann-Group/covid19_inference_forecast/blob/master/Corona_germany_simple_model.ipynb) (simple model) and [here](https://github.com/Priesemann-Group/covid19_inference_forecast/blob/master/Corona_germany_current_forecast_with3scenarios.ipynb) (with change points).
It is runnable in Google Colab. Requirement is PyMC3 >= 3.7.

Some output figures are shown below. The rest are found in the figures folder. We update them regularly.

## Modeling three different scenarios

### Summary

<img src="figures/summary_forecast.png" width="600">

### Scenario assuming three change points

<img src="figures/Fig_S3.png" width="600">
<img src="figures/Fig_S4.png" width="650">

### Scenario assuming two change points

<img src="figures/Fig_3.png" width="600">
<img src="figures/Fig_4.png" width="650">

### Scenario assuming one change point

<img src="figures/Fig_S1.png" width="600">
<img src="figures/Fig_S2.png" width="650">

