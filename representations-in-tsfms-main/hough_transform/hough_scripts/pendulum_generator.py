import numpy as np
import pandas as pd
import os
from scipy.integrate import solve_ivp

def simulate_pendulum(theta0, omega0, T=10.0, steps=512, g=9.81, L=1.0):
    def f(t, s):
        theta, omega = s
        return [omega, -(g/L) * np.sin(theta)]
    
    t_eval = np.linspace(0, T, steps)
    sol = solve_ivp(f, [0, T], [theta0, omega0], t_eval=t_eval, rtol=1e-9, atol=1e-9)
    
    theta = sol.y[0]
    omega_prime = -(g/L) * np.sin(theta)
    return theta, omega_prime

def generate_pendulum_datasets(n_series=150, length=512, g=9.81, L=1.0, output_dir="datasets_pendulum"):
    np.random.seed(42)
    os.makedirs(output_dir, exist_ok=True)

    theta_series = []
    omega_prime_series = []

    for _ in range(n_series):
        theta0 = np.random.uniform(-np.pi/2, np.pi/2)
        omega0 = np.random.uniform(-1.0, 1.0)
        theta, omega_prime = simulate_pendulum(theta0, omega0, steps=length, g=g, L=L)
        
        #print(type(omega_prime))
        #print(omega_prime.shape)
        # convert arrays to lists
        theta_series.append(theta)
        omega_prime_series.append(omega_prime)

    # enforce "series" column exactly like your working generator
    theta_df = pd.DataFrame({"series": [s for s in theta_series]})
    omega_prime_df = pd.DataFrame({"series": [s for s in omega_prime_series]})

    theta_df.to_parquet(os.path.join(output_dir, "theta.parquet"), index=False)
    omega_prime_df.to_parquet(os.path.join(output_dir, "omega_prime.parquet"), index=False)

    print(f"Saved to {output_dir}/[theta|omega_prime].parquet")

    return theta_df, omega_prime_df

if __name__ == "__main__":
    theta_df, omega_prime_df = generate_pendulum_datasets()
