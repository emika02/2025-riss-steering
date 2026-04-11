
import sys, os
sys.path.append(os.path.abspath(".."))  # go one level upimport numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import logging
from pathlib import Path
from momentfm import MOMENTPipeline
import scipy
from sklearn.decomposition import PCA
from moment import load_mean_activations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from clustering.moment import  get_activations_MOMENT
from clustering.utils import load_dataset
#from hough_scripts.separability import embeddings_pca_corr, lda_pca_embeddings, embeddings_umap
# Shape will be: (n_transform, num_samples, dimension)

num_samples=2000
model_type="moment"
output_dir="results"
device="cpu"
n_transform=10

#path = "/zfsauton2/home/ekaczmar/representations-in-tsfms-main/representations-in-tsfms-main/clustering/datasets_clusters/"
path = "/mnt/c/Users/emika/OneDrive/Pulpit/Studia 4. rok/RISS//representations-in-tsfms-main/representations-in-tsfms-main/clustering/datasets_clusters/"
path_save = "/mnt/c/Users/emika/OneDrive/Pulpit/Studia 4. rok/RISS//representations-in-tsfms-main/representations-in-tsfms-main/clustering/activations/"
#path_save = "/zfsauton2/home/ekaczmar/representations-in-tsfms-main/representations-in-tsfms-main/clustering/activations/"


activations_trends = np.stack([
    load_mean_activations(os.path.join(path, f"trends{ind+1}.parquet"), model_type, num_samples, device)
    for ind in range(n_transform)
])
np.save(os.path.join(path_save,'activations_trends.npy'), activations_trends)

activations_sines = np.stack([
    load_mean_activations(os.path.join(path, f"sines{ind+1}.parquet"), model_type, num_samples, device)
    for ind in range(n_transform)
])

np.save(os.path.join(path_save,'activations_sines.npy'), activations_sines)

activations_exps = np.stack([
    load_mean_activations(os.path.join(path, f"exps{ind+1}.parquet"), model_type, num_samples, device)
    for ind in range(n_transform)
])


np.save(os.path.join(path_save,'activations_exps.npy'), activations_exps)

