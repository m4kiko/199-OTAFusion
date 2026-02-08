import pandas as pd
import numpy as np
import os
import re
from scipy.io import savemat

# --- CONFIGURATION: Update these paths to your actual output files ---
PATHS = {
    'vit': r'D:\w\Documents\199\data\Phase1_Wired_Dataset\vit_npy_data\vit_results.csv',
    'hoc': r'D:\w\Documents\199\data\Phase1_Wired_Dataset\HOC_Classification_Results\HOC_Results.csv',
    'cyclo': r'D:\w\Documents\199\data\Phase1_Wired_Dataset\Cyclo_Classification_Results\Cyclo_Results.csv',
    'output_dir': r'D:\w\Documents\199\data\Phase1_Wired_Dataset\Consolidated'
}

# Unified modulation order for the probability matrices
MODS = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM']

def extract_frame_index(filename):
    """Extracts numeric ID from strings like '16QAM_SNR10_ID42.npy'"""
    if pd.isna(filename): return -1
    match = re.search(r'ID(\d+)', str(filename))
    return int(match.group(1)) if match else -1

def normalize_snr(series):
    """Normalize SNR to a common grid so 2.0 and 2.5 align (e.g. round to nearest 2.5)."""
    s = pd.to_numeric(series, errors='coerce')
    # Round to nearest 2.5 so ViT (0,2,5,7,10,12,15) matches HOC/Cyclo (0,2.5,5,7.5,10,12.5,15)
    return (np.round(s / 2.5) * 2.5)


def standardize_df(df, source_name):
    """Standardizes column names and casing for a specific classifier dataframe."""
    df.columns = [c.lower() for c in df.columns]
    
    # Rename probability columns to unified keys
    # Maps variants like 'prob_4psk' or 'prob_qpsk' to standard names
    rename_map = {
        'prob_bpsk': 'prob_bpsk',
        'prob_4psk': 'prob_qpsk',
        'prob_qpsk': 'prob_qpsk',
        'prob_8psk': 'prob_8psk',
        'prob_16qam': 'prob_16qam',
        'prob_64qam': 'prob_64qam',
        'true_label': 'truelabel'
    }
    df = df.rename(columns=rename_map)
    
    # Normalize SNR so different sources (e.g. 2 vs 2.5) merge correctly
    if 'snr' in df.columns:
        df['snr'] = normalize_snr(df['snr'])
    
    # Ensure truelabel is uppercase for matching
    if 'truelabel' in df.columns:
        df['truelabel'] = df['truelabel'].str.upper()
        # Fix common MATLAB to Python label naming
        df['truelabel'] = df['truelabel'].replace('4PSK', 'QPSK')

    # Ensure prediction is uppercase so MATLAB strcmp matches TrueLabel (e.g. HOC uses '8psk' -> '8PSK')
    if 'prediction' in df.columns:
        df['prediction'] = df['prediction'].astype(str).str.upper()
        df['prediction'] = df['prediction'].replace('4PSK', 'QPSK')

    return df

def main():
    if not os.path.exists(PATHS['output_dir']):
        os.makedirs(PATHS['output_dir'])

    print("Step 1: Loading and Pre-processing CSVs...")
    
    # 1. Load Data
    df_vit = pd.read_csv(PATHS['vit'])
    df_hoc = pd.read_csv(PATHS['hoc'])
    df_cyc = pd.read_csv(PATHS['cyclo'])

    # 2. Extract ID from ViT and create FrameIndex
    if 'filename' in df_vit.columns:
        df_vit['frameindex'] = df_vit['filename'].apply(extract_frame_index)

    # 2b. Normalize FrameIndex when a CSV uses global row index (e.g. 1..7000) instead of per-SNR index (1..200)
    for name, df in [('hoc', df_hoc), ('cyclo', df_cyc)]:
        fi_col = [c for c in df.columns if c.lower() == 'frameindex']
        if fi_col and df[fi_col[0]].max() > 200:
            df[fi_col[0]] = (df[fi_col[0]].astype(int) - 1) % 200 + 1

    # 3. Standardize all DataFrames
    df_vit = standardize_df(df_vit, 'vit')
    df_hoc = standardize_df(df_hoc, 'hoc')
    df_cyc = standardize_df(df_cyc, 'cyclo')

    print("Step 2: Merging Classifiers on FrameIndex and SNR...")
    
    # 4. Merge Logic (Inner Join to ensure only perfectly aligned rows remain)
    # We join on frameindex, snr, and truelabel to be 100% sure
    merge_keys = ['frameindex', 'snr', 'truelabel']
    
    merged = pd.merge(df_vit, df_hoc, on=merge_keys, suffixes=('_vit', '_hoc'))
    merged = pd.merge(merged, df_cyc, on=merge_keys, suffixes=('', '_cyc'))

    # Rename cyclo columns to have a consistent suffix
    cyc_cols = ['prediction', 'correct', 'confidence', 'prob_bpsk', 'prob_qpsk', 'prob_8psk', 'prob_16qam', 'prob_64qam']
    merged = merged.rename(columns={c: c+'_cyc' for c in cyc_cols if c in merged.columns})

    print(f"Successfully aligned {len(merged)} frames.")

    # 5. Export to CSV (Master Table)
    csv_path = os.path.join(PATHS['output_dir'], 'Consolidated_AMC_Results.csv')
    merged.to_csv(csv_path, index=False)
    print(f"[OK] Unified CSV saved: {csv_path}")

    # 6. Export to MAT (For MATLAB DS processing)
    # Constructing a struct-friendly dictionary
    mat_data = {
        'FrameIndex': merged['frameindex'].values,
        'SNR': merged['snr'].values,
        'TrueLabel': merged['truelabel'].values,
        'ViT': {
            'probs': merged[['prob_bpsk_vit', 'prob_qpsk_vit', 'prob_8psk_vit', 'prob_16qam_vit', 'prob_64qam_vit']].values,
            'pred': merged['prediction_vit'].values
        },
        'HOC': {
            'probs': merged[['prob_bpsk_hoc', 'prob_qpsk_hoc', 'prob_8psk_hoc', 'prob_16qam_hoc', 'prob_64qam_hoc']].values,
            'pred': merged['prediction_hoc'].values
        },
        'Cyclo': {
            'probs': merged[['prob_bpsk_cyc', 'prob_qpsk_cyc', 'prob_8psk_cyc', 'prob_16qam_cyc', 'prob_64qam_cyc']].values,
            'pred': merged['prediction_cyc'].values
        }
    }
    
    mat_path = os.path.join(PATHS['output_dir'], 'Consolidated_AMC_Results.mat')
    savemat(mat_path, {'results': mat_data})
    print(f"[OK] Unified MAT saved: {mat_path}")

if __name__ == "__main__":
    main()