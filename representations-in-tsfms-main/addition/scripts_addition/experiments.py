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
from .addition import compute_and_plot_separability, visualize_embeddings_pca, visualize_embeddings_lda
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


def run_addition_experiment(
    source_dataset_path, 
    target_dataset_path,
    added_dataset_path,
    model_type="moment",
    num_samples=20,
    output_dir="results",
    device="cpu",
    fit_group=1
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
    
    source_activations = extract_activations(source_dataset_path, model_type, num_samples, device)
    target_activations = extract_activations(target_dataset_path, model_type, num_samples, device)
    pre_added_activations = extract_activations(added_dataset_path, model_type, num_samples, device)
    post_added_activations = source_activations + target_activations 
    
    output_prefix = f"{source_name}_and_{target_name}"
    
    no_layers = source_activations.shape[0]
    layers_to_visualize = [0, no_layers//2, no_layers-1]
    
    for layer in layers_to_visualize:
        pca_path = os.path.join(output_dir, f"{output_prefix}_pca_layer_{layer}.pdf")
        one_reduced, other_reduced, pre_added_reduced = visualize_embeddings_pca(
            source_activations,
            target_activations,
            pre_added_activations,
            #post_added_activations,
            layer,
            title=f"Layer {layer} - PCA Visualization",
            output_file=pca_path,
            fit_group=fit_group
        )
        
        lda_path = os.path.join(output_dir, f"{output_prefix}_lda_layer_{layer}.pdf")
        one_reduced, other_reduced, pre_added_reduced = visualize_embeddings_lda(
            source_activations,
            target_activations,
            pre_added_activations,
            #post_added_activations,
            layer,
            title=f"Layer {layer} - LDA Visualization",
            output_file=lda_path
        )
    
    return one_reduced, other_reduced, pre_added_reduced

output_dir = "results"
source_dataset_path = "datasets/trend.parquet" 
target_dataset_path = "datasets/sine.parquet" 
added_dataset_path = "datasets/trend_plus_sine.parquet"

one_reduced, other_reduced, pre_added_reduced = run_addition_experiment(
    source_dataset_path, target_dataset_path, added_dataset_path, output_dir=output_dir, fit_group=1)

print("one_reduced mean:", one_reduced.mean(axis=0))
print("other_reduced mean:", other_reduced.mean(axis=0))
print("pre_added mean:", pre_added_reduced.mean(axis=0))
#print("post_added mean:", post_added_reduced.mean(axis=0)





