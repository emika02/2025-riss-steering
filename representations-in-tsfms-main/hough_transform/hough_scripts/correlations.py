import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import os
import logging
from pathlib import Path
from momentfm import MOMENTPipeline
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV #, MultiTaskLassoCV
from celer import MultiTaskLassoCV
from sklearn.metrics import mean_squared_error, r2_score



from .moment import perturb_activations_MOMENT, get_activations_MOMENT
from .chronos import perturb_activations_Chronos, get_activations_Chronos, predict_Chronos
from .perturb import add
from .steering import get_steering_matrix_and_middle_point, get_middle_point
from .utils import load_dataset, get_sample_from_dataset
from .separability import embeddings_pca_corr, embeddings_lda, embeddings_pca, lda_pca_embeddings
from .separability import compute_and_plot_separability, visualize_embeddings_pca, visualize_embeddings_lda
from .angular import cartesian_to_hyperspherical, cartesian_to_hyperspherical_batched, save_signal_plots
from .angular import hyperspherical_to_cartesian, hyperspherical_to_cartesian_batched, keep_top_n_diff, keep_top_n_diff_batched
from .angular import inject_custom_final_activations, reconstruct_signals_from_n_coord, pca_order
from .angular_plots import plot_3d_clusters, plot_angles_histogram, plot_scatter, plot_vector, plot_angles_2d


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
        
    from itertools import combinations

    def check_time_series_difference_torch(data: torch.Tensor, threshold: float, metric='l2'):
        """
        Checks if any pair of time series in the tensor differs more than the threshold.
        
        Args:
            data (torch.Tensor): shape (samples, time_points)
            threshold (float): difference threshold
            metric (str): distance metric ('l2' or 'abs')

        Returns:
            bool: True if any pair differs more than the threshold, False otherwise.
            list: List of (i, j) index pairs that exceeded the threshold.
        """
        data = data.squeeze(1)
        
        if data.dim() != 2:
            raise ValueError("Input tensor must be 2D (samples, time_points)")

        exceeding_pairs = []
        num_samples = data.shape[0]

        for i, j in combinations(range(num_samples), 2):
            x, y = data[i], data[j]
            if metric == 'l2':
                dist = torch.norm(x - y, p=2).item()
            elif metric == 'abs':
                dist = torch.max(torch.abs(x - y)).item()
            else:
                raise ValueError("Unsupported metric. Use 'l2' or 'abs'.")
            
            if dist > threshold:
                exceeding_pairs.append((i, j))

        return len(exceeding_pairs) > 0, exceeding_pairs

    '''differs, pairs = check_time_series_difference_torch(dataset, threshold=500, metric='l2')
    print("Any pair differs?", differs)
    print("Pairs that differ:", pairs)'''
            
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
    source_dataset_path, 
    target_dataset_path,
    num_samples=50,
    n_pca=[0],
    reg=None,
    model_type="moment",
    output_dir="results",
    device="cpu",
    save=False
):

    logging.info(f"Running correlation experiment: {source_dataset_path} -> {target_dataset_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    source_activations = extract_activations(source_dataset_path, model_type, num_samples, device)
    target_activations = extract_activations(target_dataset_path, model_type, num_samples, device)
    
    source_emb =  np.mean(source_activations[23, :, :, :], axis=1)
    target_emb =  np.mean(target_activations[23, :, :, :], axis=1)

    print("Dataset variance:", np.var(source_emb, axis=0).mean())
    print("Dataset linerar variance:", np.var(target_emb, axis=0).mean())
    
     #split into halves
    indices = np.random.permutation(num_samples)
    half = num_samples // 2
    train_idx, test_idx = indices[:half], indices[half:]
    
    X_train, y_train = source_emb[train_idx], target_emb[train_idx] 
    X_test, y_test = source_emb[test_idx], target_emb[test_idx]
    
    r2_l = []
    for n in n_pca:
        print("n:",n)
        if n != 0:
            source_emb_r, target_emb_r, pca = embeddings_pca_corr(X_train, y_train, n=n) 
            source_emb2_r = pca.transform(X_test)
            target_emb2_r = pca.transform(y_test)
            #source_emb_r, target_emb_r, source_emb2_r, target_emb2_r = lda_pca_embeddings(source_emb, target_emb, source_emb2, target_emb2, n_components=n)
            
        else:
            source_emb_r, target_emb_r = X_train, y_train
            source_emb2_r, target_emb2_r = X_test, y_test

        if reg == None:
            model = LinearRegression()
        elif reg == "l2":
            model = RidgeCV(alphas=[0.1, 1.0, 10.0])  # sprawdzi różne α
        elif reg == "l1":
            model = MultiTaskLassoCV(alphas=[0.001, 0.01, 0.1, 1.0], cv=5, n_jobs=-1, max_iter=10000)
            #model = MultiTaskLassoCV(alphas=[0.001, 0.01, 0.1, 1.0], cv=5, max_iter=5000)
        else:
                raise ValueError("reg must be one of: None, 'l2', 'l1'")

        # fit model
        model.fit(source_emb_r, target_emb_r)

        # predictions
        y_pred_train = model.predict(source_emb_r)
        y_pred_test  = model.predict(source_emb2_r)

        print("Linear transform")
        # evaluation
        print("Train MSE:", mean_squared_error(source_emb_r, y_pred_train))
        print("Train R^2:", r2_score(source_emb_r, y_pred_train))
        print("Test  MSE:", mean_squared_error(target_emb2_r, y_pred_test))
        print("Test  R^2:", r2_score(target_emb2_r, y_pred_test))
        
        r2_l.append(r2_score(target_emb2_r, y_pred_test))

        # jeśli model ma alpha_ (RidgeCV, LassoCV), pokaż wybrane α
        if hasattr(model, "alpha_"):
            print("Best alpha chosen:", model.alpha_)    
        
        if save==True:
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
    print(r2_l)
    

def run_correlation_experiment2(
    source_dataset_path, 
    target_dataset_path,
    next_dataset_path,
    source_dataset_path2, 
    target_dataset_path2,
    next_dataset_path2,
    num_samples=50,
    n_pca=[0], #list
    reg=None,
    model_type="moment",
    output_dir="results",
    device="cpu",
    save=False
):

    logging.info(f"Running correlation experiment: {source_dataset_path} -> {target_dataset_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(source_dataset_path2)
    source_activations = extract_activations(source_dataset_path, model_type, num_samples, device)
    target_activations = extract_activations(target_dataset_path, model_type, num_samples, device)
    next_activations = extract_activations(next_dataset_path, model_type, num_samples, device)
    
    source_activations2 = extract_activations(source_dataset_path2, model_type, num_samples, device)
    target_activations2 = extract_activations(target_dataset_path2, model_type, num_samples, device)
    next_activations2 = extract_activations(next_dataset_path2, model_type, num_samples, device)
     #source_reduced, target_reduced, next_reduced = embeddings_pca(source_activations, target_activations, target_activations, n=2)
    
    source_emb =  np.mean(source_activations[23, :, :, :], axis=1)
    target_emb =  np.mean(target_activations[23, :, :, :], axis=1)
    next_emb =  np.mean(next_activations[23, :, :, :], axis=1)
    
    source_emb2 =  np.mean(source_activations2[23, :, :, :], axis=1)
    target_emb2 =  np.mean(target_activations2[23, :, :, :], axis=1)
    next_emb2 =  np.mean(next_activations2[23, :, :, :], axis=1)
    print("Dataset variance:", np.var(source_emb, axis=0).mean())
    print("Dataset linerar variance:", np.var(target_emb, axis=0).mean())
    print("Dataset non-linerar variance:", np.var(next_emb, axis=0).mean())

    r2_l = []
    for n in n_pca:
        print("n:",n)
        if n != 0:
            source_emb_r, target_emb_r, pca = embeddings_pca_corr(source_emb, target_emb, n=n) 
            source_emb2_r = pca.transform(source_emb2)
            target_emb2_r = pca.transform(target_emb2)
            #source_emb_r, target_emb_r, source_emb2_r, target_emb2_r = lda_pca_embeddings(source_emb, target_emb, source_emb2, target_emb2, n_components=n)
            
        else:
            source_emb_r, target_emb_r = source_emb, target_emb 
            source_emb2_r, target_emb2_r = source_emb2, target_emb2
    
        
        X_train, y_train = source_emb_r, target_emb_r
        X_test, y_test = source_emb2_r, target_emb2_r

        if reg == None:
            model = LinearRegression()
        elif reg == "l2":
            model = RidgeCV(alphas=[0.1, 1.0, 10.0])  # sprawdzi różne α
        elif reg == "l1":
            model = MultiTaskLassoCV(alphas=[0.001, 0.01, 0.1, 1.0], cv=5, n_jobs=-1, max_iter=10000)
            #model = MultiTaskLassoCV(alphas=[0.001, 0.01, 0.1, 1.0], cv=5, max_iter=5000)
        else:
                raise ValueError("reg must be one of: None, 'l2', 'l1'")

        # fit model
        model.fit(X_train, y_train)

        # predictions
        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)

        print("Linear transform")
        # evaluation
        print("Train MSE:", mean_squared_error(y_train, y_pred_train))
        print("Train R^2:", r2_score(y_train, y_pred_train))
        print("Test  MSE:", mean_squared_error(y_test, y_pred_test))
        print("Test  R^2:", r2_score(y_test, y_pred_test))
        
        r2_l.append(r2_score(y_test, y_pred_test))

        # jeśli model ma alpha_ (RidgeCV, LassoCV), pokaż wybrane α
        if hasattr(model, "alpha_"):
            print("Best alpha chosen:", model.alpha_)    
        
        if save==True:
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
    
    r2_nl = []
    for n in n_pca:
        print("n:",n)
        if n != 0:
            source_emb_r, next_emb_r, pca = embeddings_pca_corr(source_emb, next_emb, n=n) 
            source_emb2_r = pca.transform(source_emb2)
            next_emb2_r = pca.transform(next_emb2)
            #source_emb_r, next_emb_r, source_emb2_r, next_emb2_r = lda_pca_embeddings(source_emb, next_emb, source_emb2, next_emb2, n_components=n)
            
        else:
            source_emb_r, next_emb_r = source_emb, next_emb
            source_emb2_r, next_emb2_r = source_emb2, next_emb2
        
        X_train, y_train = source_emb_r, next_emb_r 
        X_test, y_test = source_emb2_r, next_emb2_r

        if reg == None:
            model = LinearRegression()
        elif reg == "l2":
            model = RidgeCV(alphas=[0.1, 1.0, 10.0])  # sprawdzi różne α
        elif reg == "l1":
            model = MultiTaskLassoCV(alphas=[0.001, 0.01, 0.1, 1.0], cv=5, n_jobs=-1, max_iter=10000)
            #model = MultiTaskLassoCV(alphas=[0.001, 0.01, 0.1, 1.0], cv=5, max_iter=10000)
        else:
                raise ValueError("reg must be one of: None, 'l2', 'l1'")

        # fit model
        model.fit(X_train, y_train)

        # predictions
        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)

        print("Non-linear transform")
        # evaluation
        print("Train MSE:", mean_squared_error(y_train, y_pred_train))
        print("Train R^2:", r2_score(y_train, y_pred_train))
        print("Test  MSE:", mean_squared_error(y_test, y_pred_test))
        print("Test  R^2:", r2_score(y_test, y_pred_test))
        r2_nl.append(r2_score(y_test, y_pred_test))

        # jeśli model ma alpha_ (RidgeCV, LassoCV), pokaż wybrane α
        if hasattr(model, "alpha_"):
            
            print("Best alpha chosen:", model.alpha_)
        
        if save==True:
            coef = model.coef_
            intercept = model.intercept_
            importances = np.sum(np.abs(coef), axis=0)
            importances2 = np.sum(np.abs(coef), axis=1)

            np.save("results_corr/coef_nl.npy", importances)
            np.save("results_corr/coef_nl2.npy", importances2)
            np.save("results_corr/intercept_nl.npy", intercept)
            
            plt.figure(figsize=(12, 4))
            plt.plot(importances)
            plt.plot(importances2)
            plt.legend(["x","y"])
            plt.title("Regression coefficients (slope vector)")
            plt.xlabel("Index")
            plt.ylabel("Coefficient value")
            plt.tight_layout()
            plt.savefig("results_corr/coef_nl.png", dpi=200)
            plt.close()

            plt.figure(figsize=(12, 4))
            plt.plot(intercept.flatten())
            plt.title("Regression intercept vector")
            plt.xlabel("Index")
            plt.ylabel("Intercept value")
            plt.tight_layout()
            plt.savefig("results_corr/intercept_nl.png", dpi=200)
            plt.close()
        
    return r2_l, r2_nl
    #source_reduced, target_reduced, next_reduced = embeddings_pca(source_activations2, next_activations2, y_pred_test, n=2)
    
    #return source_reduced, target_reduced, next_reduced
    
source_dataset_path = "datasets_pendulum/pendulum_theta.parquet"
target_dataset_path = "datasets/diverse_transformed.parquet" 
next_dataset_path = "datasets/diverse_nl_transformed.parquet" 
source_dataset_path2 = "datasets2/diverse.parquet" 
target_dataset_path2 = "datasets2/diverse_transformed.parquet" 
next_dataset_path2 = "datasets2/diverse_nl_transformed.parquet"
num_samples=150
n_pca=[1,2,3,4,5,6,7,8,9, 10,25,50, 100,150, 0]
reg="l2"
model_type="moment"
output_dir="results"
device="cpu"
save=False

source_dataset_path = "datasets_pendulum/pendulum_theta.parquet"
target_dataset_path = "datasets_pendulum/pendulum_omega_prime.parquet"


run_correlation_experiment(source_dataset_path, target_dataset_path,
                   num_samples, n_pca, reg, model_type, output_dir, device)
 
'''r2_l, r2_nl = run_correlation_experiment2(source_dataset_path, target_dataset_path, next_dataset_path,
                    source_dataset_path2, target_dataset_path2, next_dataset_path2,                   
                    num_samples, n_pca, reg, model_type, output_dir, device,save)

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



