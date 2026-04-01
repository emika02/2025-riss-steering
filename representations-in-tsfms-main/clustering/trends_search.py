import numpy as np
import pandas as pd
import yaml
import logging
import os
import matplotlib.pyplot as plt
from data_generator import TimeSeriesGenerator
from moment import load_mean_activations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

def generate_trend_datasets(
    n_series=50,
    n_transform=10,
    length=512,
    max_param = 0.008,
    output_dir="datasets"):
    
    trend_series = []


    for _ in range(n_series):
        # Trend generator
        trend_gen = TimeSeriesGenerator(
            length=length,
            trend_type="linear",
            seasonality_type=None,
            noise_type=None,
            trend_params={"slope": np.random.uniform(0,max_param), "intercept": np.random.uniform(0,max_param)}, 
        )
        trend = trend_gen.generate_trend()

            
        trend_series.append(trend)
    
    datasets_trends = [trend_series,]
    
    for datasets in [datasets_trends]:
        for n in range(1, n_transform):
            datasets.append([ts**(n + 1 )  for ts in datasets[0]])
            
    os.makedirs(output_dir, exist_ok=True)

    for datasets, name in zip([datasets_trends], ["trends"]):
        for ind, dataset in enumerate(datasets):
            df = pd.DataFrame({"series": [s for s in dataset]})
            df.to_parquet(os.path.join(output_dir, name + str(ind + 1) + ".parquet"), index=False)

    print(f"Saved to {output_dir}/[trend|sine|trend_plus_sine].parquet")
    
num_samples=50
model_type="moment"
output_dir="results"
device="cpu"
n_transform=10
n_components = 2
current_output_dir = "temp_datasets"

for max_param in np.arange(0.005, 0.006, 0.0001):
    generate_trend_datasets(max_param=max_param, output_dir=current_output_dir)
    
    activations_trends = np.stack([
        load_mean_activations(os.path.join(current_output_dir, f"trends{ind+1}.parquet"), model_type, num_samples, device)
        for ind in range(n_transform)
    ])
  
    combined = activations_trends.reshape(-1, activations_trends.shape[-1])
    reducer = PCA(n_components=n_components)
    #reducer = umap.UMAP( n_components=15,n_neighbors=10,min_dist=0.0,metric="euclidean",random_state=42)

    combined_reduced = reducer.fit_transform(combined)
    n = num_samples
    reduced_list = []
    
    for i in range(n_transform):
        reduced_list.append(combined_reduced[n*i:n*(i+1)])
    
    X = np.vstack(reduced_list)
    n = reduced_list[0].shape[0]
    l = []
    for i in range(n_transform):
        l += [i] * n
    y = np.array(l)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print("max_param:" + str(max_param))
    print("Accuracy:", accuracy_score(y_test, y_pred))



    
    

