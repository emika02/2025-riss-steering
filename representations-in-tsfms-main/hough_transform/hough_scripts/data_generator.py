import numpy as np
import torch
import pandas as pd
import yaml
import logging
import os
import matplotlib.pyplot as plt


class TimeSeriesGenerator:
    def __init__(
        self,
        length=100,
        trend_type="linear",
        seasonality_type="sine",
        noise_type="gaussian",
        trend_params=None,
        seasonality_params=None,
        noise_params=None,
    ):
        self.length = length
        self.trend_type = trend_type
        self.seasonality_type = seasonality_type
        self.noise_type = noise_type
        self.trend_params = trend_params if trend_params else {}
        self.seasonality_params = seasonality_params if seasonality_params else {}
        self.noise_params = noise_params if noise_params else {}
        self.data = None
        self.trend = None
        self.seasonality = None
        self.noise = None

    def generate_trend(self):
        if self.trend_type == "linear":
            slope = self.trend_params.get("slope", 0.1)
            intercept = self.trend_params.get("intercept", 0)
            self.trend = slope * np.arange(self.length) + intercept
        elif self.trend_type == "exponential":
            growth_rate = self.trend_params.get("growth_rate", 0.01)
            self.trend = np.exp(growth_rate * np.arange(self.length))
        else:
            self.trend = None  # No trend component
            intercept = self.trend_params.get("intercept", 0)
            self.trend = intercept * np.ones(self.length)
        return self.trend

    def generate_seasonality(self):
        t = np.arange(self.length)
        if self.seasonality_type == "sine":
            amplitude = self.seasonality_params.get("amplitude", 1)
            period = self.seasonality_params.get("period", 20)
            self.seasonality = amplitude * np.sin(2 * np.pi * t / period)
        elif self.seasonality_type == "square":
            amplitude = self.seasonality_params.get("amplitude", 1)
            period = self.seasonality_params.get("period", 20)
            self.seasonality = amplitude * np.sign(np.sin(2 * np.pi * t / period))
        elif self.seasonality_type == "triangle":
            amplitude = self.seasonality_params.get("amplitude", 1)
            period = self.seasonality_params.get("period", 20)
            self.seasonality = amplitude * (
                2 * np.abs(2 * (t / period - np.floor(t / period + 0.5))) - 1
            )
        elif self.seasonality_type == "sawtooth":
            amplitude = self.seasonality_params.get("amplitude", 1)
            period = self.seasonality_params.get("period", 20)
            self.seasonality = amplitude * (
                2 * (t / period - np.floor(t / period + 0.5))
            )
        else:
            self.seasonality = None  # No seasonality component
        return self.seasonality

    def generate_noise(self):
        if self.noise_type == "gaussian":
            mean = self.noise_params.get("mean", 0)
            stddev = self.noise_params.get("stddev", 30)
            self.noise = np.random.normal(mean, stddev, self.length)
        elif self.noise_type == "uniform":
            low = self.noise_params.get("low", -1)
            high = self.noise_params.get("high", 1)
            self.noise = np.random.uniform(low, high, self.length)
        else:
            self.noise = None  # No noise component
        return self.noise

    def generate_series(self):
        trend = self.generate_trend()
        seasonality = self.generate_seasonality()
        noise = self.generate_noise()

        # Ensure we handle the case where a component is None
        self.data = np.zeros(self.length)
        if trend is not None:
            self.data += trend
        if seasonality is not None:
            self.data += seasonality
        if noise is not None:
            self.data += noise
        return self.data


def generate_trend_sine_sum_datasets(n_series=50, length=512, a=2, b=1, add_noise=False,  output_dir="datasets"):
    trend_series = []
    sine_series = []
    sum_series = []
    exp_series = []
    noise_series = []

    for _ in range(n_series):
        # Trend generator
        trend_gen = TimeSeriesGenerator(
            length=length,
            trend_type="linear",
            seasonality_type=None,
            noise_type=None,
            trend_params={"slope": np.random.uniform(5,15), "intercept": np.random.uniform(5,15)}, #(0.05,0.1)
        )
        trend = trend_gen.generate_trend()

        # Sine generator
        sine_gen = TimeSeriesGenerator(
            length=length,
            trend_type=None,
            seasonality_type="sine",
            noise_type=None,
            seasonality_params={
                "amplitude": np.random.uniform(75,125), #(25,27)
                "period": np.random.uniform(96,160), #(128,128)
            },
        )
        sine = sine_gen.generate_seasonality()
        
        exp_gen = TimeSeriesGenerator(
            length=length,
            trend_type="exponential",
            seasonality_type=None,
            noise_type=None,
            trend_params={
                "growth_rate": np.random.uniform(0.01000005,0.0100001), #(25,27)
            },
        )
        
        noise = TimeSeriesGenerator(
            length=length,
            trend_type=None,
            seasonality_type=None,
            noise_type="gaussian",
            noise_params={"mean": 0, "stdev":10}
        )
        
        noise = noise.generate_noise()
        exp = exp_gen.generate_trend()

        # Literal sum of trend and sine
        added = noise + sine
        
        if add_noise == True:
            trend += noise
            sine += noise 
            exp += noise
            

        trend_series.append(trend)
        sine_series.append(sine)
        sum_series.append(added)
        exp_series.append(exp)
        noise_series.append(noise)

        
        diverse =   trend_series + exp_series + sine_series
        diverse_transformed =  [a * ts + b for ts in diverse]
        diverse_nl_transformed =  [
        [0, 0] + [ts[i] - 2*ts[i-1] + ts[i-2] for i in range(2, len(ts))]
        for ts in diverse
        ]# [[ts[i] - ts[i-1] for i in range(1, len(ts))] for ts in diverse]

    # Convert to DataFrames
    trend_df = pd.DataFrame({"series": [s for s in trend_series]})
    sine_df = pd.DataFrame({"series": [s for s in sine_series]})
    sum_df = pd.DataFrame({"series": [s for s in sum_series]})
    exp_df = pd.DataFrame({"series": [s for s in exp_series]})
    noise_df = pd.DataFrame({"series": [s for s in noise_series]})
    diverse_df = pd.DataFrame({"series": [s for s in diverse]})
    diverse_t_df = pd.DataFrame({"series": [s for s in diverse_transformed]})
    diverse_nl_t_df = pd.DataFrame({"series": [s for s in diverse_nl_transformed]})

    # Save
    os.makedirs(output_dir, exist_ok=True)
    trend_df.to_parquet(os.path.join(output_dir, "trend.parquet"), index=False)
    sine_df.to_parquet(os.path.join(output_dir, "sine.parquet"), index=False)
    sum_df.to_parquet(os.path.join(output_dir, "trend_plus_sine.parquet"), index=False)
    exp_df.to_parquet(os.path.join(output_dir, "exp.parquet"), index=False)
    noise_df.to_parquet(os.path.join(output_dir, "noise.parquet"), index=False)
    diverse_df.to_parquet(os.path.join(output_dir, "diverse.parquet"), index=False)
    diverse_t_df.to_parquet(os.path.join(output_dir, "diverse_transformed.parquet"), index=False)
    diverse_nl_t_df.to_parquet(os.path.join(output_dir, "diverse_nl_transformed.parquet"), index=False)

    print(f"Saved to {output_dir}/[trend|sine|trend_plus_sine].parquet")

    return trend_df, sine_df, sum_df, exp_df, noise_df

n_series = 5
add_noise = False
output_dir = "datasets"
trend_df, sine_df, sum_df, exp_df, noise_df = generate_trend_sine_sum_datasets(n_series=n_series, add_noise=add_noise, output_dir=output_dir)

import os

# Create output directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Load the data
trend_df = pd.read_parquet("datasets/trend.parquet")
sine_df = pd.read_parquet("datasets/sine.parquet")
sum_df = pd.read_parquet("datasets/trend_plus_sine.parquet")
exp_df = pd.read_parquet("datasets/exp.parquet")
noise_df = pd.read_parquet("datasets/noise.parquet")
diverse_df = pd.read_parquet("datasets/diverse.parquet")
diverse_transformed_df = pd.read_parquet("datasets/diverse_transformed.parquet")
diverse_nl_transformed_df = pd.read_parquet("datasets/diverse_nl_transformed.parquet")

# Extract first time series (each row contains a full series as a list/array)
trend_series = pd.Series(trend_df.iloc[0, 0])
sine_series = pd.Series(sine_df.iloc[0, 0])
sum_series = pd.Series(sum_df.iloc[0, 0])
exp_series = pd.Series(exp_df.iloc[0, 0])
noise_series = pd.Series(noise_df.iloc[0, 0])

trend_series = pd.Series(diverse_df.iloc[0, 0])
exp_series = pd.Series(diverse_df.iloc[n_series, 0])
sine_series = pd.Series(diverse_df.iloc[2*n_series, 0])

trend_series_t = pd.Series(diverse_nl_transformed_df.iloc[0, 0])
exp_series_t = pd.Series(diverse_nl_transformed_df.iloc[n_series, 0])
sine_series_t = pd.Series(diverse_nl_transformed_df.iloc[2*n_series, 0])

# Plot and save
plt.figure(figsize=(12, 6))
'''trend_series.plot(label="Trend")
sine_series.plot(label="Sine")
exp_series.plot(label="Exp")'''
trend_series_t.plot(label="Trend t")
sine_series_t.plot(label="Sine t")
exp_series_t.plot(label="Exp t")


# Increase font sizes 2×
plt.title("Example Time Series from Each Dataset", fontsize=24)     # default ~12
plt.xlabel("Timestep", fontsize=20)                                 # default ~10
plt.ylabel("Value", fontsize=20)
plt.legend(fontsize=16)                                             # default ~8-10

plt.grid(True)
plt.tight_layout()

# Save to file instead of showing
output_path = "plots/example_time_series.png"
plt.savefig(output_path)
print(f"Plot saved to {output_path}")