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
            stddev = self.noise_params.get("stddev", 100)
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


class DiverseTimeSeriesDataset:
    def __init__(self, config_file):
        with open(config_file, "r") as file:
            self.config = yaml.safe_load(file)
        self.n_series = self.config.get("n_series", 100)
        self.length = self.config.get("length", 100)
        self.dataset = pd.DataFrame()

    def generate_diverse_dataset(self):
        for i in range(self.n_series):
            trend_type = np.random.choice(self.config["trend_types"])
            seasonality_type = np.random.choice(self.config["seasonality_types"])
            noise_type = np.random.choice(self.config["noise_types"])

            trend_params = {
                "slope": np.random.uniform(*self.config["trend_params"]["slope"]),
                "intercept": np.random.uniform(
                    *self.config["trend_params"]["intercept"]
                ),
                "growth_rate": np.random.uniform(
                    *self.config["trend_params"]["growth_rate"]
                ),
            }

            seasonality_params = {
                "amplitude": np.random.uniform(
                    *self.config["seasonality_params"]["amplitude"]
                ),
                "period": np.random.uniform(
                    *self.config["seasonality_params"]["period"]
                ),
            }

            noise_params = {
                "mean": np.random.uniform(*self.config["noise_params"]["mean"]),
                "stddev": np.random.uniform(*self.config["noise_params"]["stddev"]),
                "low": np.random.uniform(*self.config["noise_params"]["low"]),
                "high": np.random.uniform(*self.config["noise_params"]["high"]),
            }

            generator = TimeSeriesGenerator(
                length=self.length,
                trend_type=trend_type,
                seasonality_type=seasonality_type,
                noise_type=noise_type,
                trend_params=trend_params,
                seasonality_params=seasonality_params,
                noise_params=noise_params,
            )

            series_data = generator.generate_series()
            reshaped_series = series_data.flatten()

            # Creating a label dictionary for the generated series, passing None if component not present
            label = {
                "trend_type": trend_type if generator.trend is not None else "none",
                "trend_slope": (
                    trend_params["slope"] if generator.trend is not None else None
                ),
                "trend_intercept": (
                    trend_params["intercept"] if generator.trend is not None else None
                ),
                "trend_growth_rate": (
                    trend_params["growth_rate"] if generator.trend is not None else None
                ),
                "seasonality_type": (
                    seasonality_type if generator.seasonality is not None else "none"
                ),
                "seasonality_amplitude": (
                    seasonality_params["amplitude"]
                    if generator.seasonality is not None
                    else None
                ),
                "seasonality_period": (
                    seasonality_params["period"]
                    if generator.seasonality is not None
                    else None
                ),
                "noise_type": noise_type if generator.noise is not None else "none",
                "noise_mean": (
                    noise_params["mean"] if generator.noise is not None else None
                ),
                "noise_stddev": (
                    noise_params["stddev"] if generator.noise is not None else None
                ),
                "noise_low": (
                    noise_params["low"] if generator.noise is not None else None
                ),
                "noise_high": (
                    noise_params["high"] if generator.noise is not None else None
                ),
            }

            # Store the reshaped series and the labels
            series_df = pd.DataFrame({"series": [reshaped_series], **label})

            self.dataset = pd.concat([self.dataset, series_df], ignore_index=True)

        return self.dataset


def generate_and_save_dataset(config_file, dataset_name, save_to_datasets=True):
    diverse_dataset_generator = DiverseTimeSeriesDataset(config_file=config_file)
    dataset = diverse_dataset_generator.generate_diverse_dataset()
    (
        dataset.to_parquet("datasets/" + dataset_name, index=False)
        if save_to_datasets
        else None
    )

    X = np.stack(dataset["series"].values)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    logging.debug(f"X_tensor.shape: {X_tensor.shape}")
    logging.debug(f"dataset.columns: {dataset.columns}")
    logging.debug(f"dataset.head(): {dataset.head()}")

    return dataset

def generate_trend_sine_sum_datasets(n_series=512, length=512, output_dir="datasets"):
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
            trend_params={"slope": np.random.uniform(0, 2), "intercept": 0}, #(0.05,0.1)
        )
        trend = trend_gen.generate_trend()

        # Sine generator
        sine_gen = TimeSeriesGenerator(
            length=length,
            trend_type=None,
            seasonality_type="sine",
            noise_type=None,
            seasonality_params={
                "amplitude": np.random.uniform(30, 30), #(25,27)
                "period": np.random.uniform(64,128), #(128,128)
            },
        )
        sine = sine_gen.generate_seasonality()
        
        exp_gen = TimeSeriesGenerator(
            length=length,
            trend_type="exponential",
            seasonality_type=None,
            noise_type=None,
            trend_params={
                "growth_rate": np.random.uniform(0.01, 0.01000001), #(25,27)
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

        trend_series.append(trend)
        sine_series.append(sine)
        sum_series.append(added)
        exp_series.append(exp)
        noise_series.append(noise)

    # Convert to DataFrames
    trend_df = pd.DataFrame({"series": [s for s in trend_series]})
    sine_df = pd.DataFrame({"series": [s for s in sine_series]})
    sum_df = pd.DataFrame({"series": [s for s in sum_series]})
    exp_df = pd.DataFrame({"series": [s for s in exp_series]})
    noise_df = pd.DataFrame({"series": [s for s in noise_series]})

    # Save
    os.makedirs(output_dir, exist_ok=True)
    trend_df.to_parquet(os.path.join(output_dir, "trend.parquet"), index=False)
    sine_df.to_parquet(os.path.join(output_dir, "sine.parquet"), index=False)
    sum_df.to_parquet(os.path.join(output_dir, "trend_plus_sine.parquet"), index=False)
    exp_df.to_parquet(os.path.join(output_dir, "exp.parquet"), index=False)
    noise_df.to_parquet(os.path.join(output_dir, "noise.parquet"), index=False)

    print(f"Saved to {output_dir}/[trend|sine|trend_plus_sine].parquet")

    return trend_df, sine_df, sum_df, exp_df, noise_df

trend_df, sine_df, sum_df, exp_df, noise_df = generate_trend_sine_sum_datasets()

import os

# Create output directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Load the data
trend_df = pd.read_parquet("datasets/trend.parquet")
sine_df = pd.read_parquet("datasets/sine.parquet")
sum_df = pd.read_parquet("datasets/trend_plus_sine.parquet")
exp_df = pd.read_parquet("datasets/exp.parquet")
noise_df = pd.read_parquet("datasets/noise.parquet")

# Extract first time series (each row contains a full series as a list/array)
trend_series = pd.Series(trend_df.iloc[0, 0])
sine_series = pd.Series(sine_df.iloc[0, 0])
sum_series = pd.Series(sum_df.iloc[0, 0])
exp_series = pd.Series(exp_df.iloc[0, 0])
noise_series = pd.Series(noise_df.iloc[0, 0])

# Plot and save
plt.figure(figsize=(12, 6))
trend_series.plot(label="Trend")
sine_series.plot(label="Sine")
exp_series.plot(label="Exp")
noise_series.plot(label="Noise")
sum_series.plot(label="sum")

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
