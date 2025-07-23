import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import os
import logging
from pathlib import Path


from .moment import perturb_activations_MOMENT, get_activations_MOMENT
from .chronos import perturb_activations_Chronos, get_activations_Chronos, predict_Chronos
from .perturb import add
from .steering import get_steering_matrix_and_middle_point, get_middle_point
from .utils import load_dataset, get_sample_from_dataset
from .separability import compute_and_plot_separability, visualize_embeddings_pca, visualize_embeddings_lda


def cartesian_to_hyperspherical2(vector): #for 1 sample for now 
    vector = vector.detach().cpu().numpy()
    r = np.linalg.norm(vector)
    vector = vector / r #normalization
    dim = len(vector)
    angles = np.zeros(dim - 1)
    for i in range(dim - 1):
        if vector[i:].any() != 0: 
            angles[i] = np.arctan2(np.linalg.norm(vector[(i+1):]), vector[i])
        else: 
            angles[i] == 0 
    return r, angles

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
    
#can be ignored now
def calculate_angular(middle_point, source_activations, target_activations):
    middle_point = middle_point[:, np.newaxis, :, :] #shape (layers,1,...,...)
    source_differences = torch.Tensor(source_activations - middle_point)
    target_differences = torch.Tensor(target_activations - middle_point)
    source_differences = source_differences.mean(dim=2) #mean across patches
    target_differences = target_differences.mean(dim=2)
    source_mean = source_differences[23,:,:50].mean(dim=0)#mean across samples
    target_mean = target_differences[23,:,:50].mean(dim=0)
    print("source_shape:", source_mean.mean())
    print("target:", target_mean.mean())
    r_source, source_ang = cartesian_to_hyperspherical2(source_mean)
    r_target, target_ang = cartesian_to_hyperspherical2( target_mean)
    print("source:", source_ang.mean())
    print("target:", target_ang.mean())
    print("source_r:", r_source)
    print("target_r:", r_target)
    diff = source_ang - target_ang
    print("diff:", diff.mean())
    return source_differences, target_differences


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
        source_differences = torch.Tensor(source_activations - middle_point)
        target_differences = torch.Tensor(target_activations - middle_point)
        next_differences = torch.Tensor(next_activations - middle_point)
        source_differences = source_differences.mean(dim=2) #mean across patches
        target_differences = target_differences.mean(dim=2)
        next_differences = next_differences.mean(dim=2)
        source_mean = source_differences[23,:,:].mean(dim=0)#mean across samples
        target_mean = target_differences[23,:,:].mean(dim=0)
        next_mean = next_differences[23,:,:].mean(dim=0)
        print("source:", source_mean.mean())
        print("target:", target_mean.mean())
        print("next:", next_mean.mean())
        r_source, source_ang = cartesian_to_hyperspherical2(source_mean)
        r_target, target_ang = cartesian_to_hyperspherical2( target_mean)
        r_next, next_ang = cartesian_to_hyperspherical2(next_mean)
        print("source_ang:", source_ang.mean())
        print("target_ang:", target_ang.mean())
        print("next_ang:", next_ang.mean())
        print("source_r:", r_source)
        print("target_r:", r_target)
        print("next_r:", r_next)
    '''
    steering_vector = torch.cat([source_differences, target_differences], dim=1) 
    
    output_prefix = f"{source_name}_to_{target_name}_{method}_alpha_{alpha}"
    
    no_layers = source_activations.shape[0]
    layers_to_visualize = [0, no_layers//2, no_layers-1]
    
    angular_vectors = [] #np.zeros((no_layers,))
    for layer in layers_to_visualize:
        steering_vector_1 = steering_vector[layer]
        angular_vectors.append(cartesian_to_hyperspherical(steering_vector_1.clone().transpose(0,1)))
        
        if visualise == True:
            pca_path = os.path.join(output_dir, f"{output_prefix}_pca_layer_{layer}.pdf")
            visualize_embeddings_pca(
                source_activations,
                target_activations,
                layer,
                title=f"Layer {layer} - PCA Visualization",
                output_file=pca_path
            )
            
            lda_path = os.path.join(output_dir, f"{output_prefix}_lda_layer_{layer}.pdf")
            visualize_embeddings_lda(
                source_activations,
                target_activations,
                layer,
                title=f"Layer {layer} - LDA Visualization",
                output_file=lda_path
            )
    
    return np.array(angular_vectors)'''



def plot_imputed_signals_with_smoothing(imputed_normal, imputed_perturbed, window_size=10, save_path=None):
    """
    Plot imputed signals (normal and perturbed) for time series data with smoothing.
    Show non-smoothed series in pale colors and smoothed series in bold colors.
    
    Parameters:
    - imputed_normal: The normal imputed signal (numpy array).
    - imputed_perturbed: The perturbed imputed signal (numpy array).
    - window_size: The window size for the moving average smoothing.
    - save_path: If provided, saves the plot to this path as PDF.
    """
    palette = sns.color_palette()
    imputed_normal_smoothed = pd.Series(imputed_normal).rolling(window=window_size, center=True).mean()
    imputed_perturbed_smoothed = pd.Series(imputed_perturbed).rolling(window=window_size, center=True).mean()

    sns.set(font_scale=2.0, style="ticks")
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.rc('font', family='serif')

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(imputed_perturbed, label='Perturbed', color=palette[0], alpha=0.6, linewidth=2)
    ax.plot(imputed_normal, label='Non-Perturbed', color=palette[1], alpha=0.6, linewidth=2)

    ax.plot(imputed_perturbed_smoothed, label='Perturbed (Smoothed)', color=palette[0], linewidth=3)
    ax.plot(imputed_normal_smoothed, label='Non-Perturbed (Smoothed)', color=palette[1], linewidth=3)

    ax.set_xlabel("Timestep", fontsize=20)

    ax.legend(loc='best', fontsize=20)

    handles, labels = ax.get_legend_handles_labels()
    labels = [label.replace(' (Smoothed)', '') for label in labels]
    handles = [handles[2], handles[3]]
    ax.legend(handles, labels, loc='best', fontsize=20, frameon=True)
    
    plt.tight_layout(pad=2)

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')

    plt.show() 
    
source_dataset_path = "datasets/trend.parquet" 
target_dataset_path = "datasets/sine.parquet" 
next_dataset_path = "datasets/exp.parquet" 
multiple = True
model_type="moment"
method="mean"
num_samples=20
alpha=1.0
output_dir="results"
device="cpu"

run_angular_experiment(source_dataset_path, target_dataset_path, next_dataset_path, multiple,
                        model_type, method, num_samples, alpha, output_dir, device)


