import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def simulate_thermistor_series(length=512, R0=10000, T0=25+273.15, B=3950,
                               Tmin=-20, Tmax=100, mode="trend"):
    """
    Generate a thermistor time series (temperature & resistance).
    mode ∈ {"trend", "exp", "sine"}
    """
    t = np.linspace(0, 1, length)

    if mode == "trend":
        slope = np.random.uniform(0, 80)   # slope in °C over unit interval
        intercept = np.random.uniform(Tmin, Tmax)
        temp_C = slope * t + intercept

    elif mode == "exp":
        growth = np.random.uniform(0, 3)   # exponential growth/decay rate
        A = np.random.uniform(25,40)       # amplitude scale
        offset = np.random.uniform(Tmin, Tmax/2)
        temp_C = offset + A * np.exp(growth * (t - 0.5))  # centered for variety

    elif mode == "sine":
        freq = np.random.uniform(3, 5)   # frequency in cycles over [0,1]
        amp = np.random.uniform( 25,40)  # amplitude in °C
        offset = np.random.uniform(Tmin, Tmax/2)
        phase = np.random.uniform(0, 2*np.pi)
        temp_C = offset + amp * np.sin(2*np.pi*freq*t + phase)

    else:
        raise ValueError("mode must be 'trend', 'exp', or 'sine'")

    # Clip to [Tmin, Tmax] range
    temp_C = np.clip(temp_C, Tmin, Tmax)

    # resistance from Beta model
    temp_K = temp_C + 273.15
    R = R0 * np.exp(B * (1/temp_K - 1/T0))

    return temp_C, R


def generate_thermistor_datasets(n_series=150, length=512, output_dir="datasets_thermistor2"):
    np.random.seed(42)
    os.makedirs(output_dir, exist_ok=True)

    modes = ["trend", "exp", "sine"]
    n_each = n_series // len(modes)

    temp_series = []
    resist_series = []

    for mode in modes:
        for _ in range(n_each):
            temp_C, R = simulate_thermistor_series(length=length, mode=mode)
            temp_series.append(temp_C)
            resist_series.append(R)

    # Save datasets
    temp_df = pd.DataFrame({"series": [s for s in temp_series]})
    resist_df = pd.DataFrame({"series": [s for s in resist_series]})

    temp_df.to_parquet(os.path.join(output_dir, "temperature.parquet"), index=False)
    resist_df.to_parquet(os.path.join(output_dir, "resistance.parquet"), index=False)

    print(f"Saved to {output_dir}/[temperature|resistance].parquet "
          f"with {n_each} per mode, total {n_series}")

    return temp_df, resist_df


def plot_random_series(temp_df, resist_df, output_dir="plots"):
    """Pick a random time series and plot temperature & resistance, then save the plot."""
    os.makedirs(output_dir, exist_ok=True)

    idx = np.random.randint(len(temp_df))
    temp = temp_df.iloc[idx]["series"]
    resist = resist_df.iloc[idx]["series"]
    t = np.arange(len(temp))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_title(f"Thermistor Time Series (index {idx})")
    ax1.plot(t, temp, color="tab:red", label="Temperature [°C]")
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Temperature [°C]", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(t, resist, color="tab:blue", label="Resistance [Ω]")
    ax2.set_ylabel("Resistance [Ω]", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.tight_layout()

    # Save the plot
    filename = os.path.join(output_dir, f"thermistor_series_{idx}.png")
    plt.savefig(filename, dpi=150)
    print(f"Plot saved to {filename}")

    plt.show()


if __name__ == "__main__":
    temp_df, resist_df = generate_thermistor_datasets()
    plot_random_series(temp_df, resist_df)
