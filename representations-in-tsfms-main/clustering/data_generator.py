import numpy as np
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


def generate_cluster_datasets(
    n_series=50,
    n_transform=10,
    length=512,
    output_dir="datasets"):
    
    trend_series = []
    sine_series = []
    exp_series = []

    for _ in range(n_series):
        # Trend generator
        trend_gen = TimeSeriesGenerator(
            length=length,
            trend_type="linear",
            seasonality_type=None,
            noise_type=None,
            trend_params={"slope": np.random.uniform(0,0.0056), "intercept": np.random.uniform(0,0.0056)}, 
        )
        trend = trend_gen.generate_trend()

        # Sine generator
        sine_gen = TimeSeriesGenerator(
            length=length,
            trend_type=None,
            seasonality_type="sine",
            noise_type=None,
            seasonality_params={
                "amplitude": np.random.uniform(0.5,1), 
                "period": np.random.uniform(64,128), 
            },
        )
        sine = sine_gen.generate_seasonality()
        
        #Exp generator
        exp_gen = TimeSeriesGenerator(
            length=length,
            trend_type="exponential",
            seasonality_type=None,
            noise_type=None,
            trend_params={
                "growth_rate": np.random.uniform(0.002, 0.0020005 )
            },
        )
        exp = exp_gen.generate_trend()
    
            
        trend_series.append(trend)
        sine_series.append(sine)
        exp_series.append(exp)

    print("trend_var",np.mean([np.var(a) for a in trend_series]))
    print("exp_var",np.mean([np.var(a) for a in exp_series]))
    print("sine_var",np.mean([np.var(a) for a in sine_series]))
    
    datasets_trends = [trend_series,]
    datasets_sines = [sine_series,]
    datasets_exps = [exp_series,]
    
    for datasets in [datasets_trends, datasets_sines, datasets_exps]:
        for n in range(1, n_transform):
            #datasets.append([ts**(1/(n + 1) )  for ts in datasets[0]])
            datasets.append([ts**(n + 1 )  for ts in datasets[0]])
            
    os.makedirs(output_dir, exist_ok=True)

    for datasets, name in zip([datasets_trends, datasets_sines, datasets_exps], ["trends", "sines", "exps"]):
        for ind, dataset in enumerate(datasets):
            df = pd.DataFrame({"series": [s for s in dataset]})
            df.to_parquet(os.path.join(output_dir, name + str(ind + 1) + ".parquet"), index=False)

    print(f"Saved to {output_dir}/[trend|sine|trend_plus_sine].parquet")


def generate_exp_datasets(
    n_series=50,
    n_transform=10,
    length=512,
    output_dir="datasets"):
    
    exp_series = []

    for _ in range(n_series):
        
        #Exp generator
        exp_gen = TimeSeriesGenerator(
            length=length,
            trend_type="exponential",
            seasonality_type=None,
            noise_type=None,
            trend_params={
                "growth_rate": np.random.uniform(0.002, 0.0020005 )
            },
        )
        exp = exp_gen.generate_trend()

        exp_series.append(exp)

    datasets_exps = [exp_series,]
    
    for n in np.arange(1, n_transform,0.5):
        datasets_exps.append([ts**(n + 0.5 )  for ts in datasets_exps[0]])
            
    os.makedirs(output_dir, exist_ok=True)

    for ind, dataset in enumerate(datasets_exps):
        df = pd.DataFrame({"series": [s for s in dataset]})
        df.to_parquet(os.path.join(output_dir, "exp_dense" + str(ind + 1) + ".parquet"), index=False)

n_series = 200
n_transform = 10
output_dir = "datasets_clusters"
generate_cluster_datasets(n_series=n_series, output_dir=output_dir, n_transform=n_transform)
#generate_exp_datasets(n_series=n_series, output_dir=output_dir)
import os

# Create output directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Load the data
trend_df = pd.read_parquet("datasets_clusters/trends1.parquet")
sine_df = pd.read_parquet("datasets_clusters/sines1.parquet")
exp_df = pd.read_parquet("datasets_clusters/exps1.parquet")

# Extract first time series (each row contains a full series as a list/array)
trend_series = pd.Series(trend_df.iloc[0, 0])
sine_series = pd.Series(sine_df.iloc[0, 0])
exp_series = pd.Series(exp_df.iloc[0, 0])

# Plot and save
plt.figure(figsize=(12, 6))
trend_series.plot(label="Trend")
sine_series.plot(label="Sine")
exp_series.plot(label="Exp")


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