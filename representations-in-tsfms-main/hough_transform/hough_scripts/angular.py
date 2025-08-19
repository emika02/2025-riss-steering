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
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import kurtosis, skew
from .moment import create_prediction_mask_MOMENT


def cartesian_to_hyperspherical(vector): #for 1 sample for now 
    
    if isinstance(vector, torch.Tensor):
        vector = vector.detach().cpu().numpy()
        
    r = np.linalg.norm(vector)
    vector = vector / r #normalization
    dim = len(vector)
    angles = np.zeros(dim - 1)
    
    if dim == 2:
        angles[0] = np.arctan2(vector[1], vector [0])
        
    elif dim == 3:
        angles[0] = np.arccos(vector[2]/r)
        angles[1] = np.sign(vector[1]) * np.arccos(vector[0]/np.linalg.norm(vector[:-1]))
    
    else: 
        for i in range(dim - 1):
            if vector[i:].any() != 0: 
                angles[i] = np.arctan2(np.linalg.norm(vector[(i+1):]), vector[i])
            else: 
                angles[i] == 0 
    return r, angles



def hyperspherical_to_cartesian(r, angles):
    
    dim = len(angles) + 1
    cart = np.zeros(dim)
    if dim == 2:
        cart[0] =  np.cos(angles[0])
        cart[1] =  np.sin(angles[0])
        cart *= r
    
    elif dim == 3:
        cart[0] = np.sin(angles[0]) * np.cos(angles[1])
        cart[1] = np.sin(angles[0]) * np.sin(angles[1])
        cart[2] = np.cos(angles[0])
        cart *= r
        
    else:
        for i in range(dim):
            coord = r
            for j in range(i):
                coord *= np.sin(angles[j])
            if i < dim - 1:
                coord *= np.cos(angles[i])
            cart[i] = coord

    return cart


def cartesian_to_hyperspherical_batched(vectors: torch.Tensor):
    """
    Converts a batch of Cartesian vectors to hyperspherical coordinates.
    
    Args:
        vectors (torch.Tensor): shape (samples, dim)
        
    Returns:
        r:      Tensor of shape (samples,) — L2 norms (radii)
        angles: Tensor of shape (samples, dim - 1) — hyperspherical angles
    """
    
    # Convert to NumPy for easier math, if needed
    if isinstance(vectors, torch.Tensor):
        vectors_np = vectors.detach().cpu().numpy()
    else:
        vectors_np = vectors

    samples, dim = vectors_np.shape
    radii = np.linalg.norm(vectors_np, axis=1)
    
    # Normalize the vectors to avoid division errors
    safe_radii = np.where(radii == 0, 1, radii)[:, np.newaxis]
    normed_vectors = vectors_np / safe_radii

    angles = np.zeros((samples, dim - 1))

    for s in range(samples):
        v = normed_vectors[s]
        r = radii[s]

        if dim == 2:
            angles[s, 0] = np.arctan2(v[1], v[0])

        elif dim == 3:
            angles[s, 0] = np.arccos(np.clip(v[2], -1.0, 1.0))  # θ
            xy_norm = np.linalg.norm(v[:2])
            if xy_norm == 0:
                angles[s, 1] = 0
            else:
                angles[s, 1] = np.sign(v[1]) * np.arccos(np.clip(v[0] / xy_norm, -1.0, 1.0))  # φ

        else:
            for i in range(dim - 1):
                if i == dim - 2:
                    # Last angle is based on sin product
                    angles[s, i] = np.arctan2(v[i + 1], v[i])
                else:
                    denom = np.linalg.norm(v[i + 1:])
                    if denom == 0 and v[i] == 0:
                        angles[s, i] = 0
                    else:
                        angles[s, i] = np.arctan2(denom, v[i])

    return torch.tensor(radii), torch.tensor(angles)



def hyperspherical_to_cartesian_batched(radii: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    Convert batched hyperspherical coordinates to Cartesian coordinates.

    Args:
        radii: Tensor of shape (samples,)
        angles: Tensor of shape (samples, dim - 1)

    Returns:
        cart: Tensor of shape (samples, dim)
    """
    samples, dim_minus_1 = angles.shape
    dim = dim_minus_1 + 1
    cart = torch.zeros((samples, dim), dtype=radii.dtype, device=radii.device)

    if dim == 2:
        cart[:, 0] = torch.cos(angles[:, 0])
        cart[:, 1] = torch.sin(angles[:, 0])
        cart *= radii[:, None]

    elif dim == 3:
        theta = angles[:, 0]
        phi = angles[:, 1]
        cart[:, 0] = torch.sin(theta) * torch.cos(phi)
        cart[:, 1] = torch.sin(theta) * torch.sin(phi)
        cart[:, 2] = torch.cos(theta)
        cart *= radii[:, None]

    else:
        for i in range(dim):
            coord = radii.clone()
            for j in range(i):
                coord *= torch.sin(angles[:, j])
            if i < dim - 1:
                coord *= torch.cos(angles[:, i])
            cart[:, i] = coord

    return cart



def keep_top_n_diff_batched(vecs: torch.Tensor, n: int = 50, cut=True) -> torch.Tensor:
    """
    For each row (sample) in a 2D tensor, keep n entries with the largest
    absolute difference from the row mean. Set all others to pi/2.

    Args:
        vecs: torch.Tensor of shape (batch_size, dim)
        n: number of coordinates to preserve per row
        cut: if True, only consider the first 800 elements for top-n search

    Returns:
        torch.Tensor of same shape, with only top-n values kept per row
        and others set to pi/2.
    """
    if vecs.ndim != 2:
        raise ValueError("Input must be a 2D tensor")

    batch_size, dim = vecs.shape
    if n > dim:
        raise ValueError("n must be less than or equal to dim")

    mean = vecs.mean(dim=1, keepdim=True)               # (batch_size, 1)
    abs_diff = (vecs - mean).abs()                      # (batch_size, dim)

    if cut:
        # Top-k in first 800 dimensions
        if dim < 800:
            raise ValueError("Input vector must have at least 800 dimensions for cut=True")
        top_n_indices = abs_diff[:, :].topk(k=n, dim=1).indices  # (batch_size, n)
    else:
        top_n_indices = abs_diff.topk(k=n, dim=1).indices

    # Adjust indices if cut is True
    if cut:
        top_n_indices += 0  # since slicing started at 0, no offset needed

    result = torch.full_like(vecs, fill_value=np.pi / 2)
    result.scatter_(1, top_n_indices, vecs.gather(1, top_n_indices))

    return result



def keep_top_n_diff(vec, n=50, cut=True):
    """
    Keep n entries in vec with largest absolute difference from the mean,
    and set all others to pi/2.

    Args:
        vec: 1D NumPy array
        n: number of coordinates to preserve

    Returns:
        Modified copy of vec with non-top-n entries set to pi/2
    """
    if vec.ndim != 1:
        raise ValueError("Input vector must be 1D.")
    
    mean_val = np.mean(vec)
    diff = np.abs(vec - mean_val)
    
    # Get indices of top-n differences
    if cut == True: top_n_indices = np.argpartition(-diff[:800], n)[:n]
    else: top_n_indices = np.argpartition(-diff, n)[:n]
    
    # Create output vector
    result = np.full_like(vec, fill_value=np.pi / 2)
    result[top_n_indices] = vec[top_n_indices]
    
    return result

    
    
def save_signal_plots(source, target, next, title="Reconstructed mean time series"):
    
    plt.figure(figsize=(12, 6))
    plt.plot(source, label="Trend")
    plt.plot(target, label="Sine")
    plt.plot(next, label="exp")

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save to file instead of showing
    output_path = title + ".png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

 # Adjust import path as needed
from transformers.modeling_outputs import BaseModelOutput

def inject_custom_final_activations(
    filtered_activations: torch.Tensor,
    device: str = "cpu",
):
    """
    Inject pre-computed final layer activations into the MOMENT model
    and run it through the decoder to get the output.

    Args:
        filtered_activations (torch.Tensor): Tensor of shape (batch, seq_len, dim)
        device (str): 'cpu' or 'cuda'

    Returns:
        numpy.ndarray: Output from the decoder
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)

    filtered_activations = filtered_activations.to(device)
    batch_size, seq_len, dim = filtered_activations.shape

    # Create dummy dataset to satisfy input requirements
    dummy_dataset = torch.randn(batch_size, 1, seq_len).to(device)

    # Create prediction mask
    input_mask = create_prediction_mask_MOMENT(batch_size, seq_len, seq_len, device)

    # Load and initialize model
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "reconstruction", "device": device},
    )
    model.init()
    model.eval()
    model.to(device)

    # Hook to override encoder output
    def encoder_override_hook(module, input, output):
        return BaseModelOutput(last_hidden_state=filtered_activations)

    # Register hook on encoder output
    handle = model.encoder.register_forward_hook(encoder_override_hook)

    # Run model forward pass
    with torch.no_grad():
        outputs = model(x_enc=dummy_dataset, input_mask=input_mask)

    # Remove the hook after forward pass
    handle.remove()

    return outputs.reconstruction.detach().cpu().numpy()


def reconstruct_signals_from_n_coord(differences, middle_point, device, n=20, layer=23, sample_ind=0, cut=True):
    
    if isinstance(differences, np.ndarray): 
        differences = torch.Tensor(differences)
    differences = differences[layer, sample_ind,:,:]
    
    r, ang = cartesian_to_hyperspherical_batched(differences) 
    rec_source = hyperspherical_to_cartesian_batched(r, keep_top_n_diff_batched(ang, n=n, cut=cut))  
    rec_source = rec_source.unsqueeze(0)  # Add batch dimension
    sum = rec_source + middle_point
    #rec_source = inject_custom_final_activations(rec_source, device=device)
    rec_source = inject_custom_final_activations(sum[layer,:,:,:], device=device)
    
    return rec_source.flatten()


def pca_order(vector, n=None):
    # vector: (layers, samples, patches, dim)
    if n==None:
        n=dim
    dim = vector.shape[-1]

    # Step 1: Extract last layer & mean over patches -> (samples, dim)
    last_layer_mean = np.mean(vector[-1, :, :, :], axis=1)  # shape: (samples, dim)

    # Step 2: Fit PCA
    pca = PCA(n_components=n)
    pca.fit(last_layer_mean)

    # Step 3: Get rotation matrix, but reverse the order of components
    rotation = pca.components_[::-1]  # now least variance first, most variance last

    # Step 4: Apply to all layers without changing shape
    reshaped = vector.reshape(-1, dim)   # (layers*samples*patches, dim)
    reordered = reshaped @ rotation.T
    reordered = reordered.reshape(vector.shape)

    return reordered

    
    
