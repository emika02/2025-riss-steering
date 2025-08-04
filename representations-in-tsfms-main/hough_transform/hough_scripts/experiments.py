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


from .moment import perturb_activations_MOMENT, get_activations_MOMENT
from .chronos import perturb_activations_Chronos, get_activations_Chronos, predict_Chronos
from .perturb import add
from .steering import get_steering_matrix_and_middle_point, get_middle_point
from .utils import load_dataset, get_sample_from_dataset
from .separability import embeddings_pca, embeddings_lda
from .separability import compute_and_plot_separability, visualize_embeddings_pca, visualize_embeddings_lda
from .angular import cartesian_to_hyperspherical, cartesian_to_hyperspherical_batched, save_signal_plots
from .angular import hyperspherical_to_cartesian, hyperspherical_to_cartesian_batched, keep_top_n_diff, keep_top_n_diff_batched
from .angular import inject_custom_final_activations, reconstruct_signals_from_n_coord
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
    


#angular experiment for 3 clusters
def run_angular_experiment(
    source_dataset_path, 
    target_dataset_path,
    next_dataset_path,
    multiple=True, 
    model_type="moment",
    method="mean",
    num_samples=20,
    alpha=1.0,
    output_dir="results",
    device="cpu",
    visualise=False
):

    logging.info(f"Running steering experiment: {source_dataset_path} -> {target_dataset_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    source_name = Path(source_dataset_path).stem
    target_name = Path(target_dataset_path).stem
    
    source_activations = extract_activations(source_dataset_path, model_type, num_samples, device)
    target_activations = extract_activations(target_dataset_path, model_type, num_samples, device)
    next_activations = extract_activations(next_dataset_path, model_type, num_samples, device)
    middle_point = get_middle_point(source_activations, target_activations, next_activations, method=method)
    

    #can be ignored now
    if multiple == False:
        steering_vector = torch.Tensor(steering_vector).mean(dim=1)
        
    else:
        
        middle_point = middle_point[:, np.newaxis, :, :] #shape (layers,1,...,...)
        source_differences = source_activations - middle_point
        target_differences = target_activations - middle_point
        next_differences = next_activations - middle_point
    
    #source_reduced, target_reduced, next_reduced = embeddings_pca(source_differences, target_differences, next_differences, n=3)
    rec_source = reconstruct_signals_from_n_coord(source_differences, device=device, n=20, cut=True)
    rec_target = reconstruct_signals_from_n_coord(target_differences, device=device, n=20, cut=True)
    rec_next = reconstruct_signals_from_n_coord(next_differences, device=device, n=20, cut=True)
    save_signal_plots(rec_source, rec_target, rec_next)
    '''r_source, ang_source = cartesian_to_hyperspherical_batched(source_reduced)
    r_target, ang_target = cartesian_to_hyperspherical_batched(target_reduced)
    r_next, ang_next = cartesian_to_hyperspherical_batched(next_reduced)
    
    return ang_source, ang_target, ang_next,  source_reduced, target_reduced, next_reduced'''

    
source_dataset_path = "datasets/trend.parquet" 
target_dataset_path = "datasets/sine.parquet" 
next_dataset_path = "datasets/exp.parquet" 
multiple = True
model_type="moment"
method="mean"
num_samples=50
alpha=1.0
output_dir="results"
device="cpu"

ang_source, ang_target, ang_next, source_reduced, target_reduced, next_reduced = run_angular_experiment(source_dataset_path, target_dataset_path, next_dataset_path, multiple,
                        model_type, method, num_samples, alpha, output_dir, device, visualise=True)

# Plot
'''sns.set(font_scale=2.0, style="ticks")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "serif"

fig, ax = plt.subplots(figsize=(12, 10))
print(source_reduced.shape)
ax.scatter(source_reduced[:, 0], source_reduced[:, 1], c="blue", label="Trends", alpha=0.6)
ax.scatter(target_reduced[:, 0], target_reduced[:, 1], c="red", label="Sines", alpha=0.6)
ax.scatter(next_reduced[:, 0], next_reduced[:, 1], c="green", label="Exponentials", alpha=0.6)
#ax.scatter(post_added_reduced[:, 0], post_added_reduced[:, 1], c="purple", label="Post-added", alpha=0.6)
#ax.scatter(ref_reduced[:, 0], ref_reduced[:, 1], c="cyan", label="Reference", alpha=0.6)


ax.set_title("PCA", fontsize=24, pad=30)
ax.set_xlabel("Principal Component 1", fontsize=22, labelpad=20)
ax.set_ylabel("Principal Component 2", fontsize=22, labelpad=20)
ax.legend(loc="best", fontsize=18)
ax.grid(True)


plt.tight_layout()
plt.savefig("/zfsauton2/home/ekaczmar/representations-in-tsfms-main/representations-in-tsfms-main/hough_transform/vector_plots/pca.png", bbox_inches="tight")
plt.show()'''

      
plot_3d_clusters(source_reduced, target_reduced, next_reduced)      
#plot_angles_histogram(ang_source, ang_target, ang_next)
plot_angles_2d(ang_source, ang_target, ang_next)
    