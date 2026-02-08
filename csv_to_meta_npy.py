import pandas as pd
import numpy as np
import os
import sys

def convert_csv_to_meta_npy(csv_path):
    """
    Standardizes CSV outputs from Cyclo, HOC, and ViT into .npy format
    for meta-classifier fusion.
    
    Expected order in .npy: [BPSK, QPSK/4PSK, 8PSK, 16QAM, 64QAM]
    """
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    filename = os.path.basename(csv_path).lower()
    df = pd.read_csv(csv_path)

    # Define standardized target columns in order
    target_classes = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM']
    
    # 1. Column Mapping Logic
    # MATLAB (Cyclo/HOC) uses: prob_bpsk, prob_4psk, prob_8psk, prob_16qam, prob_64qam
    # Python (ViT) uses: prob_BPSK, prob_QPSK, prob_8PSK, prob_16QAM, prob_64QAM
    
    prob_cols = []
    
    if 'vit' in filename:
        print(f"Detected ViT source: {csv_path}")
        prob_cols = ['prob_BPSK', 'prob_QPSK', 'prob_8PSK', 'prob_16QAM', 'prob_64QAM']
    elif 'cyclo' in filename or 'hoc' in filename:
        print(f"Detected MATLAB source (Cyclo/HOC): {csv_path}")
        # Note: MATLAB script uses 'prob_4psk' while ViT uses 'prob_QPSK'
        # We align them here to the 5-column vector
        prob_cols = ['prob_bpsk', 'prob_4psk', 'prob_8psk', 'prob_16qam', 'prob_64qam']
    else:
        print("Error: Filename must start with or contain 'Cyclo', 'HOC', or 'vit'.")
        return

    # 2. Extract and Validate
    try:
        # Extract only the probability values
        data_matrix = df[prob_cols].values
        
        # Ensure all values are float32 for model compatibility
        data_matrix = data_matrix.astype(np.float32)
        
    except KeyError as e:
        print(f"Error: Missing columns in CSV. Expected probabilities but got: {e}")
        return

    # 3. Handle Source Prefix in Output Filename
    # Determine the source to ensure it is first in the filename as requested
    prefix = ""
    if 'vit' in filename: prefix = "vit"
    elif 'cyclo' in filename: prefix = "Cyclo"
    elif 'hoc' in filename: prefix = "HOC"
    
    # Generate output path (Same directory as input)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    # If the original filename didn't start with the prefix, add it
    if not base_name.lower().startswith(prefix.lower()):
        output_name = f"{prefix}_{base_name}.npy"
    else:
        output_name = f"{base_name}.npy"
        
    output_path = os.path.join(os.path.dirname(csv_path), output_name)

    # 4. Save
    np.save(output_path, data_matrix)
    print(f"✓ Conversion complete. Matrix shape: {data_matrix.shape}")
    print(f"✓ Saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python csv_to_meta_npy.py <path_to_csv>")
        sys.exit(1)

    convert_csv_to_meta_npy(sys.argv[1])