import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import logging
from pathlib import Path
from momentfm import MOMENTPipeline
import scipy
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV #, MultiTaskLassoCV
from celer import MultiTaskLassoCV
from sklearn.metrics import mean_squared_error, r2_score

from .moment import  get_activations_MOMENT
from .chronos import  get_activations_Chronos
from .utils import load_dataset
from .separability import embeddings_pca_corr, lda_pca_embeddings, embeddings_umap
from .angular_plots import plot_r2




def extract_activations(dataset_path, model_type="moment", num_samples=20, device="cpu"):
    """
    Extract activations from a dataset for the specified model
    
    Parameters:
    -----------
    dataset_path : str
        Path to the parquet dataset
    model_type : str
        Model type ('moment' or 'chronos')
    num_samples : int
        Number of samples to use from the dataset
    device : str
        Device to run the model on ('cpu' or 'cuda')
        
    Returns:
    --------
    activations : numpy.ndarray
        The activations extracted from the model
    """
    logging.info(f"Extracting activations from {dataset_path} using {model_type} model")
    
    dataset = load_dataset(dataset_path, type="torch", device=device)
    
    if dataset.shape[0] > num_samples:
        logging.info(f"Limiting dataset from {dataset.shape[0]} to {num_samples} samples")
        dataset = dataset[:num_samples]

            
    if model_type.lower() == "moment":
        activations = get_activations_MOMENT(dataset, device=device)
        activations = activations.cpu().numpy() if device != "cpu" else activations.numpy()
        return activations
    
    elif model_type.lower() == "chronos":
        activations_encoder, activations_decoder = get_activations_Chronos(
            dataset.squeeze(1).cpu().numpy(), device=device
        )
        activations_encoder = activations_encoder.cpu().numpy() if device != "cpu" else activations_encoder.numpy()
        activations_decoder = activations_decoder.cpu().numpy() if device != "cpu" else activations_decoder.numpy()
        return activations_encoder
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    

def run_correlation_experiment(
    source_dataset_path, #path to train input dataset
    target_dataset_path, #path to train output dataset
    source_dataset_path2, #path to test input dataset
    target_dataset_path2, #path to test output dataset
    num_samples=50,       # num_samples in each dataset
    n_pca=[0],            #list of ints, number pca of reduced dims 
    reg=None,             #type of regularization
    model_type="moment",
    output_dir="results",
    device="cpu",
    save=False
):

    logging.info(f"Running correlation experiment: {source_dataset_path} -> {target_dataset_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    source_activations = extract_activations(source_dataset_path, model_type, num_samples, device)
    target_activations = extract_activations(target_dataset_path, model_type, num_samples, device)
    
    source_activations2 = extract_activations(source_dataset_path2, model_type, num_samples, device)
    target_activations2 = extract_activations(target_dataset_path2, model_type, num_samples, device)
    
    X_train =  np.mean(source_activations[23, :, :, :], axis=1) #mean over patches, last layer
    y_train =  np.mean(target_activations[23, :, :, :], axis=1)
    X_test =  np.mean(source_activations2[23, :, :, :], axis=1)
    y_test =  np.mean(target_activations2[23, :, :, :], axis=1)

    print("Dataset variance:", np.var(X_train, axis=0).mean())
    print("Dataset transformed variance:", np.var(y_train, axis=0).mean())
    
    r2 = [] #list of r^2 scores for different number of reduced dims
    for n in n_pca:
        print("Dim number:",n)
        if n != 0:
            source_emb_r, target_emb_r, reducer = embeddings_umap(X_train, y_train, n=n) 
            source_emb2_r = reducer.transform(X_test)
            target_emb2_r = reducer.transform(y_test)
            #source_emb_r, target_emb_r, source_emb2_r, target_emb2_r = lda_pca_embeddings(source_emb, target_emb, source_emb2, target_emb2, n_components=n)
            
        else: #if n==0 there's no dimensionality reduction
            source_emb_r, target_emb_r = X_train, y_train
            source_emb2_r, target_emb2_r = X_test, y_test

        if reg == None:
            model = LinearRegression()
        elif reg == "l2":
            model = RidgeCV(alphas=[0.1, 1.0, 10.0])  
        elif reg == "l1":
            model = MultiTaskLassoCV(alphas=[0.001, 0.01, 0.1, 1.0], cv=5, n_jobs=-1, max_iter=10000)
        else:
                raise ValueError("reg must be one of: None, 'l2', 'l1'")

        # fit model
        model.fit(source_emb_r, target_emb_r)

        # predictions
        y_pred_train = model.predict(source_emb_r)
        y_pred_test  = model.predict(source_emb2_r)
        
        '''mean_trends = target_emb2_r[:50,:].mean(axis=0)
        mean_exps = target_emb2_r[50:100,:].mean(axis=0)
        mean_sines = target_emb2_r[100:,:].mean(axis=0)
        
        y_pred_test = np.zeros(target_emb2_r.shape)
        y_pred_test[:50] = mean_trends
        y_pred_test[50:100] = mean_exps
        y_pred_test[100:] = mean_sines'''
                
        # evaluation
        print("Train MSE:", mean_squared_error(target_emb_r, y_pred_train))
        print("Train R^2:", r2_score(target_emb_r, y_pred_train))
        print("Test  MSE:", mean_squared_error(target_emb2_r, y_pred_test))
        print("Test  R^2:", r2_score(target_emb2_r, y_pred_test))
        
        r2.append(r2_score(target_emb2_r, y_pred_test))

        # jeśli model ma alpha_ (RidgeCV, LassoCV), pokaż wybrane α
        if hasattr(model, "alpha_"):
            print("Best alpha chosen:", model.alpha_)    
        
        if save==True: #saving linear regression coefficients
            coef = model.coef_
            intercept = model.intercept_
            importances = np.sum(np.abs(coef), axis=0)
            importances2 = np.sum(np.abs(coef), axis=1)

            np.save("results_corr/coef_linear.npy", importances)
            np.save("results_corr/coef_linear2.npy", importances2)
            np.save("results_corr/intercept_linear.npy", intercept)
            
            
            plt.figure(figsize=(12, 4))
            plt.plot(importances)
            plt.plot(importances2)
            plt.legend(["x","y"])
            plt.title("Regression coefficients (slope vector)")
            plt.xlabel("Index")
            plt.ylabel("Coefficient value")
            plt.tight_layout()
            plt.savefig("results_corr/coef_linear.png", dpi=200)
            plt.close()

            plt.figure(figsize=(12, 4))
            plt.plot(intercept.flatten())
            plt.title("Regression intercept vector")
            plt.xlabel("Index")
            plt.ylabel("Intercept value")
            plt.tight_layout()
            plt.savefig("results_corr/intercept_linear.png", dpi=200)
            plt.close()
    print(r2)
    return r2
    
num_samples=150
n_pca=[1,5,10,25,50, 100,150,200,500,0]
reg="l2"
model_type="moment"
output_dir="results"
device="cpu"
save=False

source_dataset_path = "datasets_pendulum/theta.parquet"
target_dataset_path = "datasets_pendulum/omega_prime.parquet"
source_dataset_path2 = "datasets_pendulum2/theta.parquet"
target_dataset_path2 = "datasets_pendulum2/omega_prime.parquet"

source_dataset_path = "datasets/diverse.parquet"
target_dataset_path = "datasets/diverse_nl_transformed.parquet"
source_dataset_path2 = "datasets2/diverse.parquet"
target_dataset_path2 = "datasets2/diverse_nl_transformed.parquet"

'''source_dataset_path = "datasets_thermistor/temperature.parquet"
target_dataset_path = "datasets_thermistor/resistance.parquet"
source_dataset_path2 = "datasets_thermistor2/temperature.parquet"
target_dataset_path2 = "datasets_thermistor2/resistance.parquet"'''



r2 = run_correlation_experiment(source_dataset_path, target_dataset_path,
                           source_dataset_path2, target_dataset_path2,
                   num_samples, n_pca, reg, model_type, output_dir, device)

#Plot r2 scores vs n_pca, saved to results_corr/pca_curve.png
plot_r2(n_pca, r2)
'''
fig, ax = plt.subplots(figsize=(12, 10))
ax.scatter(source_reduced[:, 0], source_reduced[:, 1], c="blue", label="Test Dataset", alpha=0.6)
ax.scatter(target_reduced[:, 0], target_reduced[:, 1], c="red", label="Test Dataset Transformed original", alpha=0.6)
ax.scatter(next_reduced[:, 0], next_reduced[:, 1], c="green", label="Test Dataset Transformed Reconstructed", alpha=0.6)


# Set titles and labels (2x size) 
ax.set_title("PCA", fontsize=30, pad=30)
ax.set_xlabel("Principal Component 1", fontsize=26, labelpad=20)
ax.set_ylabel("Principal Component 2", fontsize=26, labelpad=20)
ax.legend(loc="best", fontsize=22)
ax.grid(True)

# Save and show
plt.tight_layout()
plt.savefig("/zfsauton2/home/ekaczmar/representations-in-tsfms-main/representations-in-tsfms-main/hough_transform/results_corr/pca.png", bbox_inches="tight")
plt.show()'''



