# Improved Online Conformal Prediction via Strongly Adaptive Online Learning
This library implements numerous algorithms which perform conformal prediction on data with arbitrary distribution
shifts over time. This is the official implementation for the [paper](https://arxiv.org/abs/2302.07869) Bhatnagar et al.,
"Improved Online Conformal Prediction via Strongly Adaptive Online Learning," 2023. We include reference implementations
for the proposed methods Strongly Adaptive Online Conformal Prediction (SAOCP) and Scale-Free Online Gradient Descent
(SF-OGD), as well as Split Conformal Prediction ([Vovk et al., 1999](https://dl.acm.org/doi/10.5555/645528.657641)),
Non-Exchangeable Conformal Prediction ([Barber et al., 2022](https://arxiv.org/pdf/2202.13415.pdf)), and 
Fully Adaptive Conformal Inference (FACI, [Gibbs & Candes, 2022](https://arxiv.org/pdf/2208.08401.pdf)).

## Replicating Our Experiments
First install the `online_conformal` package by cloning this repo and calling ``pip install .``.
To run our time series forecasting experiments, first clone the [Merlion](https://github.com/salesforce/Merlion) repo
and install their `ts_datasets` package. Then, you can call
```shell
python time_series.py --model <model> --dataset <dataset> --njobs <njobs>
```
where `<model>` can be one of `LGBM`, `ARIMA`, or `Prophet`; `<dataset>` can be one of
`M4_Hourly`, `M4_Daily`, `M4_Weekly`, or `NN5_Daily`; and `<njobs>` indicates the number of 
parallel cores you wish to parallelize the file with. The results will be written to a sub-directory
`results`.

To run our experiments on image classification under distribution shift, first install [PyTorch](https://pytorch.org/).
Then, you can call
```shell
python vision.py --dataset <dataset>
```
where dataset is one of `ImageNet` or `TinyImageNet`. Various intermediate results will be written to
sub-folders, and checkpointing (e.g. for model training) is automatic.

## Using Our Code
To use our code, first install the `online_conformal` package by calling ``pip install online_conformal``. 
You can alternatively install the package from source by cloning this repo and calling ``pip install .``.

Each online conformal prediction method is implemented as its own class in the package. All methods share a common API.
For time series forecasting, we leverage models implemented in [Merlion](https://github.com/salesforce/Merlion).
Below, we demonstrate how to use `SAOCP` to create prediction intervals for multi-horizon time series forecasting.
The update loop is a simplified version of calling
`saocp.forecast(time_series=test_data.iloc[:horizon], time_series_prev=train_data)`, whose implementation you can find
[here](https://github.com/salesforce/online_conformal/blob/main/online_conformal/base.py#L116).

```python
import pandas as pd
from merlion.models.factory import ModelFactory
from merlion.utils import TimeSeries
from online_conformal.dataset import M4
from online_conformal.saocp import SAOCP

# Get some time series data as pandas.DataFrames
data = M4("Hourly")[0]
train_data, test_data = data["train_data"], data["test_data"]
# Initialize a Merlion model for time series forecasting
model = ModelFactory.create(name="LGBMForecaster")
# Initialize the SAOCP wrapper on top of the model. This splits the data 
# into train/calibration splits, trains the model on the train split, 
# and initializes SAOCP's internal state on the calibration split.
# The target coverage is 90% here, but you can adjust this freely.
# We also do 24-step-ahead forecasting by setting horizon=24.
horizon = 24
saocp = SAOCP(model=model, train_data=train_data, coverage=0.9,
              calib_frac=0.2, horizon=horizon)

# Get the model's 24-step-ahead prediction, and convert it to prediction intervals
yhat, _ = saocp.model.forecast(horizon, time_series_prev=TimeSeries.from_pd(train_data))
delta_lb, delta_ub = zip(*[saocp.predict(horizon=h + 1) for h in range(horizon)])
yhat = yhat.to_pd().iloc[:, 0]
lb, ub = yhat + delta_lb, yhat + delta_ub

# Update SAOCP's internal state based on the next 24 observations
prev = train_data.iloc[:-horizon + 1]
time_series = pd.concat((train_data.iloc[-horizon + 1:], test_data.iloc[:horizon]))
for i in range(len(time_series)):
    # Predict yhat_{t-H+i+1}, ..., yhat_{t-H+i+H} = f(y_1, ..., y_{t-H+i}) 
    y = time_series.iloc[i:i + horizon, 0]
    yhat, _ = saocp.model.forecast(y.index, time_series_prev=TimeSeries.from_pd(prev))
    yhat = yhat.to_pd().iloc[:, 0]
    # Use h-step prediction of yhat_{t-k+h} to update SAOCP's h-step prediction interval
    for h in range(len(y)):
        if i >= h:
            saocp.update(ground_truth=y[h:h + 1], forecast=yhat[h:h + 1], horizon=h + 1)
    prev = pd.concat((prev, time_series.iloc[i:i+1]))
```

For other use cases, you can initialize `saocp = SAOCP(model=None, train_data=None, max_scale=max_scale, coverage=0.9)`.
Here, `max_scale` indicates the largest value you expect the conformal score to take. Then, you can obtain the conformal
score corresponding to 90% (or your desired level of coverage) by calling `score = saocp.predict(horizon=1)[1]`, and
you can use this value to compute the prediction set `{y: S(X_t, y) < score}` using your own custom code. Finally, after
you observe the true conformal score `new_score = S(X_t, Y_t)`, you can update the conformal predictor by calling 
`saocp.update(ground_truth=pd.Series([new_score]), forecast=pd.Series([0]), horizon=1)`. 
