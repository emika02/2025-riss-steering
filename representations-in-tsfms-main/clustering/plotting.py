
import numpy as np
import matplotlib.pyplot as plt 

def plot_knn(X_test, y_test, n_transform, powers=None):
    
    '''plots 2d points from test set with ground truth labels'''
    
    if powers == None:
        powers = powers = np.arange(1, int(n_transform + 1), 1)
        
    # Slice the 5D test set down to 2D for the plot
    X_test_2d = X_test[:, :2] 

    # Filter X_test using the TRUE labels (y_test) to create separate groups for the scatter
    results_list = [X_test_2d[y_test == i] for i in range(n_transform)]

    # Generate automated labels and 19 distinct colors
    labels = [f"$x^{{{p:.1f}}}$" for p in powers]
    labels[0] = "Input ($x^{1.0}$)"
    colors = plt.cm.turbo(np.linspace(0, 1, n_transform)) # 'turbo' is great for 19 gradients

    # 7. Plotting
    fig, ax = plt.subplots(figsize=(14, 8))

    for reduced, color, label in zip(results_list, colors, labels):
        ax.scatter(reduced[:, 0], reduced[:, 1], 
                color=color, label=label, alpha=0.7, s=120, edgecolors='white', linewidth=0.5)

    # Styling
    ax.set_title(f"PCA Visualization of KNN Test Set ({n_transform} Power Transformations)", fontsize=22, pad=20)
    ax.set_xlabel("Principal Component 1", fontsize=16)
    ax.set_ylabel("Principal Component 2", fontsize=16)

    # Move legend outside because 19 classes take up a lot of vertical space
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Transformation", fontsize=10, ncol=1)
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()
    

def plot_knn_3d(X_test, y_test, n_transform, powers=None):
    
    '''plots 3d points from test set with ground truth labels'''
    
    if powers == None:
        powers = powers = np.arange(1, int(n_transform + 1), 1)
        
    # Slice the 5D test set down to 2D for the plot
    X_test_2d = X_test[:, :3] 

    # Filter X_test using the TRUE labels (y_test) to create separate groups for the scatter
    results_list = [X_test_2d[y_test == i] for i in range(n_transform)]

    # Generate automated labels and 19 distinct colors
    labels = [f"$x^{{{p:.1f}}}$" for p in powers]
    labels[0] = "Input ($x^{1.0}$)"
    colors = plt.cm.turbo(np.linspace(0, 1, n_transform)) # 'turbo' is great for 19 gradients

    # 7. Plotting
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d') # Create 3D axes

    for reduced, color, label in zip(results_list, colors, labels):
        # Scatter requires x, y, and z coordinates
        ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2],
                color=color, label=label, alpha=0.6, s=60, edgecolors='white', linewidth=0.3)

    # Styling for 3D
    ax.set_title(f"3D PCA trends ({n_transform} Power Transformations)", fontsize=20, pad=30)
    ax.set_xlabel("PC 1", fontsize=14)
    ax.set_ylabel("PC 2", fontsize=14)
    ax.set_zlabel("PC 3", fontsize=14)

    # Adjust viewing angle for better perspective
    ax.view_init(elev=20, azim=45)
    # Legend outside
    ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5), title="Transformation", fontsize=9, ncol=1)

    plt.tight_layout()
    plt.show()

    