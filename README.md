# Bayesian inference and forecast of COVID-19

[![Documentation Status](https://readthedocs.org/projects/covid19-inference-forecast/badge/?version=latest)](https://covid19-inference-forecast.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

We want to quantify the effect of new policies on the spread of COVID-19. Crucially, fitting an exponential function to the number of cases lacks an interpretability of the fitting error. We built a Bayesian SIR model where we can incorporate our prior knowledge of the time points of governmental policy changes. At the example of Germany, we show that the two kinks in the last weeks correspond to two changes of policies, leading to a growth rate of about 0 now.

The research article [is available on arXiv](https://arxiv.org/abs/2004.01105).

The code used to produce the figures is available [here](https://github.com/Priesemann-Group/covid19_inference_forecast/blob/master/Corona_germany_simple_model.ipynb) (simple model) and [here](https://github.com/Priesemann-Group/covid19_inference_forecast/blob/master/scripts/SIR_with_delay_Germany_3scenarios.ipynb) (with change points).
It is runnable in Google Colab. Requirement is PyMC3 >= 3.7.

If you want to use the code, we recommend to look at our [documentation](https://covid19-inference-forecast.readthedocs.io/en/latest/).

Some output figures are shown below. The rest are found in the figures folder. We update them regularly.

### Please take notice of our [disclaimer](disclaimer.md).

## Modeling three different scenarios in Germany

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

