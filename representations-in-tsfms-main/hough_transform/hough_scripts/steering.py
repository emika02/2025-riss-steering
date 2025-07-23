from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from tqdm import tqdm

def compute_lda_steering_vector(one_data, other_data):
    """
    Compute the vector pointing from one class to another along the LDA discriminant direction.
    """
    X = np.concatenate((one_data, other_data), axis=0)

    y = np.concatenate((np.zeros(one_data.shape[0]), np.ones(other_data.shape[0])))

    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

    lda_direction = lda.coef_.flatten()

    return lda_direction


def get_steering_matrix_and_middle_point(one_activations, other_activations, method="median", n_jobs=-1):
    """
    Get the steering matrix from one_activations to other_activations.
    Parallelized for 'lda' method, with progress tracking.
    """
    layer, batch, patch, features = one_activations.shape

    if method == "median":
        one_median = np.median(one_activations, axis=1)
        other_median = np.median(other_activations, axis=1)
        steering_matrix = other_median - one_median
        middle_point = one_median + 0.5 * steering_matrix

    elif method == "mean":
        one_mean = np.mean(one_activations, axis=1)
        other_mean = np.mean(other_activations, axis=1)
        steering_matrix = other_mean - one_mean
        middle_point = one_mean + 0.5 * steering_matrix

    elif method == "lda": #middle point not implemented
        steering_matrix = np.zeros((layer, patch, features))

        def compute_for_patch(l, p):
            one_data = one_activations[l, :, p, :]
            other_data = other_activations[l, :, p, :]
            return l, p, compute_lda_steering_vector(one_data, other_data)

        tasks = [(l, p) for l in range(layer) for p in range(patch)]

        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_for_patch)(l, p) for l, p in tqdm(tasks, desc="Computing LDA Steering Vectors")
        )

        for l, p, lda_vector in results:
            steering_matrix[l, p, :] = lda_vector

    return steering_matrix, middle_point 

def get_middle_point(one_activations, other_activations, next_activations, method="median", n_jobs=-1):


    if method == "median":
        one_median = np.median(one_activations, axis=1)
        other_median = np.median(other_activations, axis=1)
        next_median = np.median(next_activations, axis=1)
        middle_point = (one_median + other_median + next_median)/3

    elif method == "mean":
        one_mean = np.mean(one_activations, axis=1)
        other_mean = np.mean(other_activations, axis=1)
        next_mean = np.mean(next_activations, axis=1)
        middle_point = (one_mean + other_mean + next_mean)/3

    return  middle_point 