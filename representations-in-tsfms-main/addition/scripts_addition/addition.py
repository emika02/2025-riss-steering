import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from itertools import product
from tqdm import tqdm


def compute_linear_separability(
    layer, patch, activations_class_one, activations_class_other, no_samples
):
    """
    Computes the Fisher's Linear Discriminant Ratio for a given layer and patch.

    Parameters:
    layer: int, current layer index
    patch: int, current patch index
    activations_class_one: numpy array, activations for class one (sine_constant)
    activations_class_other: numpy array, activations for class other (none_constant)
    no_samples: int, number of samples

    Returns:
    tuple: layer, patch, and Fisher's score
    """
    activations = np.concatenate(
        (activations_class_one, activations_class_other), axis=0
    )
    labels = np.concatenate((np.ones(no_samples), np.zeros(no_samples)))

    lda = LinearDiscriminantAnalysis()
    lda.fit(activations, labels)

    projections = lda.transform(activations)
    projections_class_one = projections[:no_samples]
    projections_class_other = projections[no_samples:]

    mean1 = projections_class_one.mean()
    mean2 = projections_class_other.mean()
    var1 = projections_class_one.var()
    var2 = projections_class_other.var()

    fisher_score = ((mean1 - mean2) ** 2) / (var1 + var2)

    return layer, patch, fisher_score


def plot_linear_separability(
    linear_separability,
    linear_separability_mean,
    no_layers,
    fontsize=40,
    output_file="linear_separability_heatmap.pdf",
):
    """
    Plots a heatmap of linear separability and an overlay of mean separability across layers.

    Parameters:
    linear_separability: 2D numpy array for heatmap data (shape: patches x layers)
    linear_separability_mean: 1D numpy array of mean separability for each layer
    no_layers: int, number of layers (x-axis length)
    title: str, title for the plot
    output_file: str, the file name to save the plot
    """

    sns.set(font_scale=2.0, style="ticks")
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"

    fig, ax1 = plt.subplots(figsize=(12, 10))

    fig.subplots_adjust(right=0.85, top=0.92, bottom=0.1, left=0.1)

    cbar_ax = fig.add_axes([1.0, 0.2, 0.05, 0.7])
    heatmap = sns.heatmap(
        linear_separability.T, cmap="viridis", ax=ax1, cbar=True, cbar_ax=cbar_ax
    )
    cbar_ax.tick_params(labelsize=fontsize)

    ax1.set_xlabel("Model Depth", fontsize=fontsize, labelpad=20)
    ax1.set_ylabel("Patch Position", fontsize=fontsize, labelpad=20)
    ax1.tick_params(axis="y", labelsize=fontsize-2, colors='black')
    ax1.tick_params(axis="x", labelsize=fontsize-2, colors='black')
    
    ax2 = ax1.twinx()
    ax2.plot(
        np.arange(no_layers),
        linear_separability_mean,
        color="red",
        linewidth=3,
        label="Scaled LDR",
    )
    ax2.tick_params(axis="y", labelsize=fontsize-2, colors='red')
    ax2.set_xlim(0, no_layers - 1)
    ax2.set_ylabel("Scaled LDR", fontsize=fontsize, labelpad=20, color='red')

    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=0)

    xticklabels = ax1.get_xticklabels()
    yticklabels = ax1.get_yticklabels()

    xtick_positions = ax1.get_xticks()
    new_xticklabels = [
        label.get_text() if i % 4 == 0 else ''  # Keep every 2nd label
        for i, label in enumerate(xticklabels)
    ]
    ax1.set_xticklabels(new_xticklabels, rotation=0)

    ytick_positions = ax1.get_yticks()
    new_yticklabels = [
        label.get_text() if i % 4 == 0 else ''  # Keep every 2nd label
        for i, label in enumerate(yticklabels)
    ]
    ax1.set_yticklabels(new_yticklabels, rotation=0)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    print(f"Plot saved as {output_file}")

def visualize_embeddings_pca(
    one_activations,
    other_activations,
    pre_added_activations,
    post_added_activations,
    ref_activations,
    coordinates,
    title="Layer Embeddings - PCA with Shifted Samples",
    output_file="embedding_visualization.pdf",
    fit_group=1,
):
    """
    Visualize the embeddings in a selected layer and patch after applying PCA 
    and highlight separability between sine and none samples.

    Parameters:
    sine_constant_activations: numpy array, activations for sine_constant input
    none_constant_activations: numpy array, activations for none_constant input
    layer_to_visualize: int, the layer index to visualize
    patch: int, patch index to visualize
    title: str, the title for the plot
    output_file: str, the file name to save the plot
    """
    
    '''print(one_activations[0].mean()) 
    print(other_activations[0].mean())
    print(pre_added_activations[0].mean()) '''

    
    if isinstance(coordinates, tuple):
        layer_to_visualize, patch = coordinates
        one_patch_embeddings = one_activations[layer_to_visualize, :, patch, :]
        other_patch_embeddings = other_activations[layer_to_visualize, :, patch, :]
        pre_added_embeddings = pre_added_activations[layer_to_visualize, :, patch, :]
        post_added_embeddings = post_added_activations[layer_to_visualize, :, patch, :]
        ref_embeddings = ref_activations[layer_to_visualize, :, patch, :]
        
    else:
        layer_to_visualize = coordinates
        one_patch_embeddings = np.mean(one_activations[layer_to_visualize, :, :, :], axis=1)
        other_patch_embeddings = np.mean(other_activations[layer_to_visualize, :, :, :], axis=1)
        pre_added_embeddings = np.mean(pre_added_activations[layer_to_visualize, :, :, :], axis=1)
        post_added_embeddings = np.mean(post_added_activations[layer_to_visualize, :, :, :], axis=1)
        ref_embeddings = np.mean(ref_activations[layer_to_visualize, :, :, :], axis=1)
    '''print(one_patch_embeddings[0].mean()) 
    print(other_patch_embeddings[0].mean())
    print(pre_added_embeddings[0].mean()) '''

    if fit_group == 2:
        combined = np.concatenate([one_patch_embeddings, other_patch_embeddings, pre_added_embeddings], axis=0)
        pca = PCA(n_components=2)
        combined_reduced = pca.fit_transform(combined)
        post_added_reduced = pca.transform(post_added_embeddings)
        ref_reduced = pca.transform(ref_embeddings)
        
        n_source = one_patch_embeddings.shape[0]
        n_target = other_patch_embeddings.shape[0]
        n_pre_added = pre_added_embeddings.shape[0]
        #n_post_added = post_added_embeddings.shape[0]

        idx1 = n_source
        idx2 = idx1 + n_target
        #idx3 = idx2 + n_pre_added
        # idx4 = idx3 + n_post_added  # Not needed unless debugging

        source_reduced = combined_reduced[:idx1]
        target_reduced = combined_reduced[idx1:idx2]
        pre_added_reduced = combined_reduced[idx2:]
        #post_added_reduced = combined_reduced[idx3:]
        
    elif fit_group == 1:
        
        combined = np.concatenate([one_patch_embeddings, other_patch_embeddings], axis=0)
        pca = PCA(n_components=2)
        combined_reduced = pca.fit_transform(combined)
        pre_added_reduced = pca.transform(pre_added_embeddings)
        post_added_reduced = pca.transform(post_added_embeddings)
        ref_reduced = pca.transform(ref_embeddings)
        
        idx1 = one_patch_embeddings.shape[0]

        source_reduced = combined_reduced[:idx1]
        target_reduced = combined_reduced[idx1:]
        
    elif fit_group == 0:
        pca = PCA(n_components=2)
        source_reduced = pca.fit_transform(one_patch_embeddings)
        target_reduced = pca.transform(other_patch_embeddings)
        pre_added_reduced = pca.transform(pre_added_embeddings)
        post_added_reduced = pca.transform(post_added_embeddings)
        ref_reduced = pca.transform(ref_embeddings)

    else:
        raise Exception("fit_group values can only be 0, 1, 2")
    
    post_added_reduced = source_reduced + target_reduced
    # Plot
    sns.set(font_scale=2.0, style="ticks")
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(source_reduced[:, 0], source_reduced[:, 1], c="blue", label="Source", alpha=0.6)
    ax.scatter(target_reduced[:, 0], target_reduced[:, 1], c="red", label="Target", alpha=0.6)
    ax.scatter(pre_added_reduced[:, 0], pre_added_reduced[:, 1], c="green", label="Pre-added", alpha=0.6)
    ax.scatter(post_added_reduced[:, 0], post_added_reduced[:, 1], c="purple", label="Post-added", alpha=0.6)
    #ax.scatter(ref_reduced[:, 0], ref_reduced[:, 1], c="cyan", label="Reference", alpha=0.6)

    
    ax.set_title(title, fontsize=24, pad=30)
    ax.set_xlabel("Principal Component 1", fontsize=22, labelpad=20)
    ax.set_ylabel("Principal Component 2", fontsize=22, labelpad=20)
    ax.legend(loc="best", fontsize=18)
    ax.grid(True)
    
    for i in range(len(pre_added_reduced)):
        ax.plot(
            [pre_added_reduced[i, 0], post_added_reduced[i, 0]],
            [pre_added_reduced[i, 1], post_added_reduced[i, 1]],
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.6
        )

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    print(f"Embedding visualization saved as {output_file}")
    return source_reduced, target_reduced, pre_added_reduced, post_added_reduced, ref_reduced
    
def visualize_embeddings_lda(
    one_activations,
    other_activations,
    pre_added_activations,
    post_added_activations,
    ref_activations,
    coordinates,
    title="Layer Embeddings - LDA with Shifted Samples",
    output_file="embedding_visualization_lda_1d.pdf",
    two_components=True
):
    """
    Visualize the embeddings in a selected layer and patch after applying LDA 
    and highlight separability between two classes (Class 0 and Class 1).
    This version reduces the embeddings to one dimension.

    Parameters:
    one_activations: numpy array, activations for Class 0 input
    other_activations: numpy array, activations for Class 1 input
    coordinates: tuple or int, layer and patch indices or just layer index for visualization
    title: str, the title for the plot
    output_file: str, the file name to save the plot
    """
    
    # Extract embeddings based on the provided coordinates
    if isinstance(coordinates, tuple):
        layer_to_visualize, patch = coordinates
        one_patch_embeddings = one_activations[layer_to_visualize, :, patch, :]
        other_patch_embeddings = other_activations[layer_to_visualize, :, patch, :]
        pre_added_patch_embeddings = pre_added_activations[layer_to_visualize, :, patch, :]
        post_added_patch_embeddings = post_added_activations[layer_to_visualize, :, patch, :]
        ref_patch_embeddings = ref_activations[layer_to_visualize, :, patch, :]
        
    else:
        layer_to_visualize = coordinates
        one_patch_embeddings = np.mean(one_activations[layer_to_visualize, :, :, :], axis=1)
        other_patch_embeddings = np.mean(other_activations[layer_to_visualize, :, :, :], axis=1)
        pre_added_patch_embeddings = np.mean(pre_added_activations[layer_to_visualize, :, :, :], axis=1) 
        post_added_patch_embeddings = np.mean(post_added_activations[layer_to_visualize, :, :, :], axis=1)
        ref_patch_embeddings = np.mean(ref_activations[layer_to_visualize, :, :, :], axis=1)
        
    if two_components:
        labels = np.concatenate([
        np.zeros(one_patch_embeddings.shape[0]),
        np.ones(other_patch_embeddings.shape[0]),
        np.full(pre_added_patch_embeddings.shape[0], 2)
        ])    
        
        combined_embeddings = np.vstack([
        one_patch_embeddings,
        other_patch_embeddings,
        pre_added_patch_embeddings
        ])
        
        lda = LinearDiscriminantAnalysis(n_components=2)
        reduced_embeddings = lda.fit_transform(combined_embeddings, labels)
        post_added_reduced = lda.transform(post_added_patch_embeddings)
        ref_reduced = lda.transform(ref_patch_embeddings)

        n0 = one_patch_embeddings.shape[0]
        n1 = other_patch_embeddings.shape[0]
        one_reduced = reduced_embeddings[:n0]
        other_reduced = reduced_embeddings[n0:n0 + n1]
        pre_added_reduced = reduced_embeddings[n0 + n1:]
        
        post_added_reduced = one_reduced + other_reduced

        sns.set(font_scale=2.0, style="ticks")
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["font.family"] = "serif"

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(one_reduced[:, 0], one_reduced[:, 1], c="blue", label="Class 0", alpha=0.6)
        ax.scatter(other_reduced[:, 0], other_reduced[:, 1], c="red", label="Class 1", alpha=0.6)
        ax.scatter(pre_added_reduced[:, 0], pre_added_reduced[:, 1], c="green", label="Pre-added", alpha=0.6)
        ax.scatter(post_added_reduced[:, 0], post_added_reduced[:, 1], c="purple", label="Post-added", alpha=0.6)
        #ax.scatter(ref_reduced[:, 0], ref_reduced[:, 1], c="cyan", label="Reference", alpha=0.6)

        
        ax.set_title(title, fontsize=24, pad=30)
        ax.set_xlabel("LDA Component 1", fontsize=20)
        ax.set_ylabel("LDA Component 2", fontsize=20)
        ax.legend(loc="best", fontsize=16)
        ax.grid(True)
        
        for i in range(len(pre_added_reduced)):
            ax.plot(
                [pre_added_reduced[i, 0], post_added_reduced[i, 0]],
                [pre_added_reduced[i, 1], post_added_reduced[i, 1]],
                color="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.6
            )

        plt.tight_layout()
        plt.savefig(output_file, bbox_inches="tight")
        plt.show()
        print(f"Embedding visualization saved as {output_file}")
        
        return one_reduced, other_reduced, pre_added_reduced, post_added_reduced, ref_reduced
    else:       
        labels = np.concatenate([np.zeros(one_patch_embeddings.shape[0]), np.ones(other_patch_embeddings.shape[0])])
        
        combined_embeddings = np.vstack([one_patch_embeddings, other_patch_embeddings])

        lda = LinearDiscriminantAnalysis(n_components=1)
        reduced_embeddings = lda.fit_transform(combined_embeddings, labels)
        reduced_pre_added = lda.transform(pre_added_patch_embeddings)
        reduced_post_added = lda.transform(post_added_patch_embeddings)
        ref_reduced = lda.transform(ref_patch_embeddings)
        #reduced_post_added = lda.transform(post_added_activations)

        one_reduced = reduced_embeddings[labels == 0]
        other_reduced = reduced_embeddings[labels == 1]
        
        post_added_reduced = one_reduced + other_reduced

        # Plot
        sns.set(font_scale=2.0, style="ticks")
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["font.family"] = "serif"

        fig, ax1 = plt.subplots(figsize=(14, 6))

        ax1.scatter(one_reduced[:, 0], np.zeros_like(one_reduced[:, 0]), c="blue", label="Class 0", alpha=0.6)
        ax1.scatter(other_reduced[:, 0], np.zeros_like(other_reduced[:, 0]), c="red", label="Class 1", alpha=0.6)
        ax1.scatter(reduced_pre_added[:, 0], np.full_like(reduced_pre_added[:, 0], 0.1), c="green", label="Pre-added", alpha=0.6)
        ax1.scatter(reduced_post_added[:, 0], np.full_like(reduced_post_added[:, 0], -0.1), c="purple", label="Post-added", alpha=0.6)
        #ax1.scatter(ref_reduced[:, 0], np.full_like(ref_reduced[:, 0], -0.1), c="cyan", label="Reference", alpha=0.6)

        ax1.set_title(title, fontsize=24, pad=30)
        ax1.set_xlabel("LDA Component 1", fontsize=22, labelpad=20)
        ax1.set_yticks([])
        ax1.legend(loc="best", fontsize=18)
        ax1.grid(True)

        plt.tight_layout()
        pre_y = 0.1
        post_y = -0.1
        
        # Connect each pre-added and post-added point
        for i in range(len(reduced_pre_added)):
            ax1.plot(
                [reduced_pre_added[i, 0], reduced_post_added[i, 0]],
                [pre_y, post_y],
                color="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.6
            )

        
        plt.savefig(output_file, bbox_inches="tight")
        plt.show()
        print(f"Embedding visualization saved as {output_file}")
        return one_reduced, other_reduced, reduced_pre_added, reduced_post_added, ref_reduced

def compute_and_plot_separability(
    one_activations,
    other_activations,
    prefix="default_model/",
):
    """
    Main function to compute linear separability and generate the plot, and
    visualize PCA for highest and lowest separability.

    Parameters:
    sine_constant_activations: numpy array, activations for sine_constant input
    none_constant_activations: numpy array, activations for none_constant input
    output_file: str, name of the output image file for the separability heatmap
    embedding_high_output_file: str, name of the output image file for the PCA visualization of highest separability
    embedding_low_output_file: str, name of the output image file for the PCA visualization of lowest separability
    """

    no_layers, no_samples, no_patches, no_features = one_activations.shape

    linear_separability = np.zeros((no_layers, no_patches))

    layer_patch_combinations = list(product(range(no_layers), range(no_patches)))

    results = Parallel(n_jobs=-1)(
        delayed(compute_linear_separability)(
            layer,
            patch,
            one_activations[layer, :, patch, :],
            other_activations[layer, :, patch, :],
            no_samples,
        )
        for layer, patch in tqdm(layer_patch_combinations, desc="Processing layers and patches")
    )

    for layer_idx, patch_idx, score in results:
        linear_separability[layer_idx, patch_idx] = score

    linear_separability = (linear_separability - linear_separability.min()) / (
        linear_separability.max() - linear_separability.min()
    )

    sine_average = np.mean(one_activations, axis=2, keepdims=True)
    none_average = np.mean(other_activations, axis=2, keepdims=True)

    linear_separability_mean = np.zeros((no_layers))

    results_mean = Parallel(n_jobs=-1)(
        delayed(compute_linear_separability)(
            layer,
            0,
            sine_average[layer, :, 0, :],
            none_average[layer, :, 0, :],
            no_samples,
        )
        for layer in range(no_layers)
    )

    for layer_idx, _, score in results_mean:
        linear_separability_mean[layer_idx] = score

    linear_separability_mean = (
        linear_separability_mean - linear_separability_mean.min()
    ) / (linear_separability_mean.max() - linear_separability_mean.min())

    plot_linear_separability(
        linear_separability,
        linear_separability_mean,
        no_layers,
        output_file=prefix + "linear_separability_heatmap.pdf",
    )

    patch_highest = np.unravel_index(
        np.argmax(linear_separability, axis=None), linear_separability.shape
    )
    patch_lowest = np.unravel_index(
        np.argmin(linear_separability, axis=None), linear_separability.shape
    )
    
    layer_mean_higest = np.argmax(linear_separability_mean)
    layer_mean_lowest = np.argmin(linear_separability_mean)

    visualize_embeddings_pca(
        one_activations,
        other_activations,
        patch_highest,
        title=f"Layer {patch_highest[0]} - Patch {patch_highest[1]}, Highest Separability",
        output_file=prefix + "embedding_visualization_pca_high.pdf",
    )

    visualize_embeddings_pca(
        one_activations,
        other_activations,
        patch_lowest,
        title=f"Layer {patch_lowest[0]} - Patch {patch_lowest[1]}, Lowest Separability",
        output_file=prefix + "embedding_visualization_pca_low.pdf",
    )
    
    visualize_embeddings_pca(
        one_activations,
        other_activations,
        layer_mean_higest,
        title=f"Layer {layer_mean_higest} - Mean, Highest Separability",
        output_file=prefix + "embedding_visualization_pca_high_mean.pdf",
    )
    
    visualize_embeddings_pca(
        one_activations,
        other_activations,
        layer_mean_lowest,
        title=f"Layer {layer_mean_lowest} - Mean, Lowest Separability",
        output_file=prefix + "embedding_visualization_pca_low_mean.pdf",
    )
    
    visualize_embeddings_lda(
        one_activations,
        other_activations,
        patch_highest,
        title=f"Layer {patch_highest[0]} - Patch {patch_highest[1]}, Highest Separability",
        output_file=prefix + "embedding_visualization_lda_high.pdf",
    )
    
    visualize_embeddings_lda(   
        one_activations,
        other_activations,
        patch_lowest,
        title=f"Layer {patch_lowest[0]} - Patch {patch_lowest[1]}, Lowest Separability",
        output_file=prefix + "embedding_visualization_lda_low.pdf",
    )
    
    visualize_embeddings_lda(
        one_activations,
        other_activations,
        layer_mean_higest,
        title=f"Layer {layer_mean_higest} - Mean, Highest Separability",
        output_file=prefix + "embedding_visualization_lda_high_mean.pdf",
    )
    
    visualize_embeddings_lda(
        one_activations,
        other_activations,
        layer_mean_lowest,
        title=f"Layer {layer_mean_lowest} - Mean, Lowest Separability",
        output_file=prefix + "embedding_visualization_lda_low_mean.pdf",
    )


if __name__ == "__main__":

    sine_constant_activations = np.load("activations/sine_constant_activations.npy")[
        :, :256, :, :
    ]
    none_constant_activations = np.load("activations/none_constant_activations.npy")[
        :, :256, :, :
    ]

    compute_and_plot_separability(
        sine_constant_activations,
        none_constant_activations,
        prefix="default_model/",
    )
