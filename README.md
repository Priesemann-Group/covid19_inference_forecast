# Covid-19 inference and forecast

We attempt to model the spread of Covid-19. 

The research article [is available on arXiv](https://arxiv.org/abs/2004.01105).

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

