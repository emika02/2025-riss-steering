import sys
import os
import torch
import numpy as np
import logging
from pathlib import Path
from momentfm import MOMENTPipeline
from moment import load_mean_activations_batched

# Ensure local imports work depending on your structure
sys.path.append(os.path.abspath(".."))
from clustering.utils import load_dataset 

# --- CONFIGURATION ---
NUM_SAMPLES = 2000
MODEL_TYPE = "moment"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TRANSFORM = 10
BATCH_SIZE = 32  # Adjust based on GPU VRAM (16 or 32 is usually safe)
LAYER_IDX = 23   # The last layer of MOMENT-1-large

# Paths - Update these to your cluster environment
#PATH_DATA = "/mnt/c/Users/emika/OneDrive/Pulpit/Studia 4. rok/RISS//representations-in-tsfms-main/representations-in-tsfms-main/clustering/datasets_clusters/"
#PATH_SAVE = "/mnt/c/Users/emika/OneDrive/Pulpit/Studia 4. rok/RISS//representations-in-tsfms-main/representations-in-tsfms-main/clustering/activations/"
PATH_DATA = "/zfsauton2/home/ekaczmar/representations-in-tsfms-main/representations-in-tsfms-main/clustering/datasets_clusters/"
PATH_SAVE = "/zfsauton2/home/ekaczmar/representations-in-tsfms-main/representations-in-tsfms-main/clustering/activations/"

logging.basicConfig(level=logging.INFO)


# --- MAIN EXECUTION ---

def main():
    os.makedirs(PATH_SAVE, exist_ok=True)
    
    print(f"Initializing MOMENT model on {DEVICE}...")
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "reconstruction", "device": DEVICE},
    )
    model.init()
    model.eval()
    model.to(DEVICE)

    datasets = ["trends", "sines", "exps"]

    for name in datasets:
        print(f"\n--- Processing Dataset: {name} ---")
        
        # 1. Prepare Memory-Mapped file to prevent RAM explosion
        # Shape: (10 transforms, 2000 samples, 1024 dimensions)
        mmap_path = os.path.join(PATH_SAVE, f'activations_{name}.dat')
        final_array = np.memmap(
            mmap_path, 
            dtype='float32', 
            mode='w+', 
            shape=(N_TRANSFORM, NUM_SAMPLES, 1024)
        )

        for ind in range(N_TRANSFORM):
            file_path = os.path.join(PATH_DATA, f"{name}{ind+1}.parquet")
            
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found. Skipping.")
                continue
                
            print(f"Computing activations for {name} transform {ind+1}...")
            
            # 2. Extract and pool (returns 2000x1024)
            mean_act = load_mean_activations_batched(file_path, model, MODEL_TYPE, NUM_SAMPLES, DEVICE, layer=LAYER_IDX)
            
            # 3. Write directly to disk
            final_array[ind] = mean_act.astype('float32')
            final_array.flush()
            
            # 4. Explicit cleanup for this iteration
            del mean_act

        print(f"Successfully saved {name} to {mmap_path}")

    print("\nAll tasks completed successfully.")

if __name__ == "__main__":
    main()