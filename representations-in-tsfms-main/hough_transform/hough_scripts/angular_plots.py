import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting
import numpy as np
import torch
import seaborn as sns 
import os

def plot_3d_clusters(source_reduced, target_reduced, next_reduced):
    """
    Plots three clusters of 3D points with labels: trends, sines, exponentials.
    
    Args:
        source_reduced (np.ndarray): Array of shape (N, 3)
        target_reduced (np.ndarray): Array of shape (N, 3)
        next_reduced (np.ndarray): Array of shape (N, 3)
        save_path (str): Optional path to save the figure.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each cluster
    ax.scatter(source_reduced[:, 0], source_reduced[:, 1], source_reduced[:, 2], 
               label="trends", c='blue', alpha=0.7)
    
    ax.scatter(target_reduced[:, 0], target_reduced[:, 1], target_reduced[:, 2], 
               label="sines", c='green', alpha=0.7)
    
    ax.scatter(next_reduced[:, 0], next_reduced[:, 1], next_reduced[:, 2], 
               label="exponentials", c='red', alpha=0.7)

    # Labels
    ax.set_title("3D Clusters: Trends, Sines, Exponentials")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # Show or save
    
    plt.savefig("/zfsauton2/home/ekaczmar/representations-in-tsfms-main/representations-in-tsfms-main/hough_transform/vector_plots/3d_clusters.png", bbox_inches='tight')

    
def plot_angles_2d(ang_source, ang_target, ang_next):
    sns.set(font_scale=2.0, style="ticks")
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(ang_source[:, 0], ang_source[:, 1], c="blue", label="Trends", alpha=0.6)
    ax.scatter(ang_target[:, 0], ang_target[:, 1], c="red", label="Sines", alpha=0.6)
    ax.scatter(ang_next[:, 0], ang_next[:, 1], c="green", label="Exponentials", alpha=0.6)


    ax.set_title("Angular components", fontsize=24, pad=30)
    ax.set_xlabel("Theta", fontsize=22, labelpad=20)
    ax.set_ylabel("Phi", fontsize=22, labelpad=20)
    ax.legend(loc="best", fontsize=18)
    ax.grid(True)


    plt.tight_layout()
    plt.savefig("/zfsauton2/home/ekaczmar/representations-in-tsfms-main/representations-in-tsfms-main/hough_transform/vector_plots/theta_phi.png", bbox_inches="tight")
    plt.show()
    
    
def plot_mutual_angular_distance_histogram(source_vecs, target_vecs, next_vecs, save_dir="vector_plots", filename="true_angular_hist.png"):
    os.makedirs(save_dir, exist_ok=True)

    # Ensure everything is numpy
    if isinstance(source_vecs, torch.Tensor):
        source_vecs = source_vecs.detach().cpu().numpy()
    if isinstance(target_vecs, torch.Tensor):
        target_vecs = target_vecs.detach().cpu().numpy()
    if isinstance(next_vecs, torch.Tensor):
        next_vecs = next_vecs.detach().cpu().numpy()

    # Select a random reference from source
    ref = source_vecs[np.random.randint(source_vecs.shape[0])]  # shape (dim,)
    ref = ref / np.linalg.norm(ref)  # just in case, normalize to unit

    # Concatenate all vectors
    all_vecs = np.concatenate([source_vecs, target_vecs, next_vecs], axis=0)
    all_vecs = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)  # normalize all

    # Compute angular distances
    dot_products = np.clip(np.dot(all_vecs, ref), -1.0, 1.0)
    angles = np.arccos(dot_products)  # shape (N,)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.hist(angles, bins=75, color='purple', alpha=0.7)
    plt.title("Angular Distances to Random Source Vector (on Hypersphere)")
    plt.xlabel("Angular Distance (radians)")
    plt.ylabel("Frequency")
    plt.grid(True)

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    
def plot_angles_histogram(source_vecs, target_vecs, next_vecs, save_dir="vector_plots", filename="angles_1d_hist_lda.png"):
    os.makedirs(save_dir, exist_ok=True)

    # Ensure everything is numpy
    if isinstance(source_vecs, torch.Tensor):
        source_vecs = source_vecs.detach().cpu().numpy()
    if isinstance(target_vecs, torch.Tensor):
        target_vecs = target_vecs.detach().cpu().numpy()
    if isinstance(next_vecs, torch.Tensor):
        next_vecs = next_vecs.detach().cpu().numpy()

    
    all_vecs = np.concatenate([source_vecs, target_vecs, next_vecs], axis=0)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.hist(all_vecs, bins=75, color='purple', alpha=0.7)
    plt.title("Angular Coordinates - 1d")
    plt.xlabel("Angular Distance (radians)")
    plt.ylabel("Frequency")
    plt.grid(True)

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    
    
def plot_separate_angular_distance_histograms(source_vecs, target_vecs, next_vecs, save_dir="vector_plots"):
    os.makedirs(save_dir, exist_ok=True)

    # Convert to numpy if tensors
    def to_numpy(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

    source_vecs = to_numpy(source_vecs)
    target_vecs = to_numpy(target_vecs)
    next_vecs   = to_numpy(next_vecs)

    # Random reference vector from source, normalized
    ref = source_vecs[np.random.randint(source_vecs.shape[0])]
    ref = ref / np.linalg.norm(ref)

    # Normalize all input vectors
    def normalize(v):
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    source_vecs = normalize(source_vecs)
    target_vecs = normalize(target_vecs)
    next_vecs   = normalize(next_vecs)

    # Angular distances = arccos(dot) with clipping
    def angular_distances(vectors, ref):
        dots = np.clip(np.dot(vectors, ref), -1.0, 1.0)
        return np.arccos(dots)

    cluster_data = {
        'source': angular_distances(source_vecs, ref),
        'target': angular_distances(target_vecs, ref),
        'next':   angular_distances(next_vecs,   ref),
    }

    # Create and save one histogram per cluster
    for name, distances in cluster_data.items():
        plt.figure(figsize=(8, 4))
        plt.hist(distances, bins=75, color='purple', alpha=0.7)
        plt.title(f"Angular Distances to Random Source Vector ({name})")
        plt.xlabel("Angular Distance (radians)")
        plt.ylabel("Frequency")
        plt.grid(True)
        save_path = os.path.join(save_dir, f"angular_hist_{name}.png")
        plt.savefig(save_path)
        plt.close()

def plot_scatter(vec, vec1, vec2, save_dir="vector_plots", title="Vector Plot"):
    os.makedirs(save_dir, exist_ok=True)

    vec = np.array(vec.flatten())
    vec1 = np.array(vec1.flatten())
    vec2 = np.array(vec2.flatten())
    indices = np.arange(len(vec)) 
    print(len(vec))# index-based coloring

    def save_scatter(x, y, xlabel, ylabel, filename):
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(x, y, c=indices, cmap='viridis', s=20, alpha=0.8)
        plt.colorbar(scatter, label="Index")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        plt.close()

    save_scatter(vec, vec1, "trends", "sines", "Trends_vs_Sines.png")
    save_scatter(vec1, vec2, "sines", "exponential", "Sines_vs_Exponential.png")
    save_scatter(vec, vec2, "trends", "exponentials", "Trends_vs_Exponentials.png")
    
    
def plot_vector(vec, save_dir="vector_plots", title="Vector Plot"):
    os.makedirs(save_dir, exist_ok=True)
    
    vec = np.array(vec)
    
    plt.figure(figsize=(10, 4))
    plt.plot(vec, marker='.', markersize=2, linewidth=1)
    plt.title(title)
    plt.ylim((1.2,2.5))
    plt.xlabel("Dimension")
    plt.ylabel("Value")
    plt.axhline(y=np.pi/2, color='red', linestyle='-', linewidth=1)
    plt.grid(True)
    
    save_path = os.path.join(save_dir, title + ".png")
    plt.savefig(save_path)
    plt.close()