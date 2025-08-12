import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from pathlib import Path

from .moment import perturb_activations_MOMENT, get_activations_MOMENT
from .chronos import perturb_activations_Chronos, get_activations_Chronos, predict_Chronos
#from .perturb import add
#from .steering import get_steering_matrix
from .utils import load_dataset, get_sample_from_dataset
from.perturb import add
from .addition import compute_and_plot_separability, visualize_embeddings_pca, get_steering_matrix, visualize_embeddings_lda
from .data_generator import generate_trend_sine_sum_datasets

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

def plot_imputed_signals_with_smoothing(imputed_normal, imputed_normal_org, imputed_perturbed, window_size=10, save_path="/zfsauton2/home/ekaczmar/representations-in-tsfms-main/representations-in-tsfms-main/addition/results/imputed"):
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
    imputed_normal_smoothed_org = pd.Series(imputed_normal_org).rolling(window=window_size, center=True).mean()
    imputed_perturbed_smoothed = pd.Series(imputed_perturbed).rolling(window=window_size, center=True).mean()

    sns.set(font_scale=2.0, style="ticks")
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.rc('font', family='serif')

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(imputed_perturbed, label='Perturbed', color=palette[0], alpha=0.6, linewidth=2)
    ax.plot(imputed_normal, label='Non-Perturbed', color=palette[1], alpha=0.6, linewidth=2)
    ax.plot(imputed_normal_org, label='Original', color=palette[2], alpha=0.6, linewidth=2)

    ax.plot(imputed_perturbed_smoothed, label='Perturbed (Smoothed)', color=palette[0], linewidth=3)
    ax.plot(imputed_normal_smoothed, label='Non-Perturbed (Smoothed)', color=palette[1], linewidth=3)
    ax.plot(imputed_normal_smoothed_org, label='Original (Smoothed)', color=palette[2], linewidth=3)

    ax.set_xlabel("Timestep", fontsize=20)

    ax.legend(loc='best', fontsize=20)
    
    handles, labels = ax.get_legend_handles_labels()

    # Remove the "(Smoothed)" text
    labels = [label.replace(' (Smoothed)', '') for label in labels]

    # Deduplicate while keeping order
    seen = {}
    final_handles = []
    final_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            final_handles.append(h)
            final_labels.append(l)

    ax.legend(final_handles, final_labels, loc='best', fontsize=20, frameon=True)
    
    plt.tight_layout(pad=2)

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')

    plt.show() 

def run_addition_experiment(
    source_dataset_path, 
    target_dataset_path,
    added_dataset_path,
    reference_dataset_path,
    model_type="moment",
    num_samples=20,
    output_dir="results",
    device="cpu",
    fit_group=1,
    two_components=True
):
    """
    Run a addition experiment
    
    Parameters:
    -----------
    source_dataset_path : str
        Path to the source dataset (parquet)
    target_dataset_path : str
        Path to the target dataset (parquet)
    model_type : str
        Model type ('moment' or 'chronos')
    num_samples : int
        Number of samples from each dataset
    output_dir : str
        Directory to save results
    device : str
        Device to run the model on ('cpu' or 'cuda')
        
    Returns:
    --------
    dict
        Dictionary containing the results of the experiment
    """
    logging.info(f"Running addition experiment: {source_dataset_path} -> {target_dataset_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    source_name = Path(source_dataset_path).stem
    target_name = Path(target_dataset_path).stem
    output_prefix = f"{source_name}_and_{target_name}"
    
    source_activations = extract_activations(source_dataset_path, model_type, num_samples, device)
    target_activations = extract_activations(target_dataset_path, model_type, num_samples, device)
    pre_added_activations = extract_activations(added_dataset_path, model_type, num_samples, device)
    
    ref_activations = extract_activations(reference_dataset_path, model_type, num_samples, device)
    post_added_activations = source_activations + target_activations 
    
    steering_vector = get_steering_matrix( source_activations, target_activations, method="median")
    
    input_dataset = pd.read_parquet(added_dataset_path)
    input_sample = get_sample_from_dataset(input_dataset, 5)
    
    input_org_sample = get_sample_from_dataset(pd.read_parquet(target_dataset_path), 5)
    
    if model_type.lower() == "moment":
        non_perturbed_output = perturb_activations_MOMENT(input_sample, device=device)
        non_perturbed_output_org = perturb_activations_MOMENT(input_org_sample, device=device)
        perturbed_output = perturb_activations_MOMENT(
            input_sample, 
            perturbation_fn=add, 
            perturbation_payload= 0.9 * steering_vector,
            device=device
        )
        non_perturbed_output = non_perturbed_output.flatten()
        non_perturbed_output_org = non_perturbed_output_org.flatten()
        perturbed_output = perturbed_output.flatten()
    
    elif model_type.lower() == "chronos":
        input_sample_np = input_sample.squeeze(1).cpu().numpy()
        non_perturbed_output = predict_Chronos(
            input_sample_np, 
            prediction_length=64,
            device=device
        )
        
        perturbed_output = perturb_activations_Chronos(
            input_sample_np[:,np.newaxis,:],
            prediction_length=64,
            device=device,
            perturbation_fn=add,
            perturbation_payload=0.5 * steering_vector
        )
        
        non_perturbed_output = non_perturbed_output.cpu().numpy().flatten()
        perturbed_output = perturbed_output.cpu().numpy().flatten()
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    plot_path = os.path.join(output_dir, f"{output_prefix}.pdf")
    plot_imputed_signals_with_smoothing(non_perturbed_output, non_perturbed_output_org, perturbed_output, save_path=plot_path)

    
    no_layers = source_activations.shape[0]
    layers_to_visualize = [0, no_layers//2, no_layers-1]
    
    for layer in layers_to_visualize:
        pca_path = os.path.join(output_dir, f"{output_prefix}_pca_layer_{layer}_fit_group_{fit_group}.pdf")
        one_reduced, other_reduced, pre_added_reduced, post_added_reduced, ref_reduced = visualize_embeddings_pca(
            source_activations,
            target_activations,
            pre_added_activations,
            post_added_activations,
            ref_activations,
            layer,
            title=f"Layer {layer} - PCA Visualization",
            output_file=pca_path,
            fit_group=fit_group
        )
        
        lda_path = os.path.join(output_dir, f"{output_prefix}_lda_layer_{layer}_2d_{two_components}.pdf")
        one_reduced, other_reduced, pre_added_reduced, post_added_reduced, ref_reduced = visualize_embeddings_lda(
            source_activations,
            target_activations,
            pre_added_activations,
            post_added_activations,
            ref_activations,
            layer,
            title=f"Layer {layer} - LDA Visualization",
            output_file=lda_path,
            two_components=two_components
        )
    
    return one_reduced, other_reduced, pre_added_reduced, post_added_reduced, ref_reduced

output_dir = "results"
source_dataset_path = "datasets/noise.parquet" 
target_dataset_path = "datasets/sine.parquet" 
added_dataset_path = "datasets/trend_plus_sine.parquet"
reference_dataset_path = "datasets/exp.parquet"

one_reduced, other_reduced, pre_added_reduced, post_added_reduced, ref_reduced = run_addition_experiment(
    source_dataset_path, target_dataset_path, added_dataset_path, reference_dataset_path,
    output_dir=output_dir, fit_group=2, two_components=False)

print("one_reduced mean:", one_reduced.mean(axis=0))
print("other_reduced mean:", other_reduced.mean(axis=0))
print("pre_added mean:", pre_added_reduced.mean(axis=0))
print("post_added mean:", post_added_reduced.mean(axis=0))






