import numpy as np
import pandas as pd
import os
import sys
import re
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

# --- USER CONFIGURATION: Point to your individual CSV result files ---
# Using CSVs is more robust than NPYs because we can sort by FrameIndex
VIT_CSV_PATH   = r'D:\w\Documents\199\data\Phase1_Wired_Dataset\vit_npy_data\vit_results.csv'
HOC_CSV_PATH   = r'D:\w\Documents\199\data\Phase1_Wired_Dataset\HOC_Classification_Results\HOC_Results.csv'
CYCLO_CSV_PATH = r'D:\w\Documents\199\data\Phase1_Wired_Dataset\Cyclo_Classification_Results\Cyclo_Results.csv'

# Output location
OUTPUT_DIR     = r'D:\w\Documents\199\data\Phase1_Wired_Dataset\DS_Fusion_Results'

# DS Parameters
UNCERTAINTY_THRESHOLD = 0.05
CONFLICT_THRESHOLD = 0.95
FUSION_CLASSES = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM']
# -----------------------------------------------------------------

def calculate_reliability(snr, zc_metric=None):
    """
    Step 0: Reliability Calibration (Context-Aware Weighting).
    Uses SNR-based accuracy lookup and optional ZC peak scaling.
    """
    zc_weight = 1.0
    if zc_metric is not None and not pd.isna(zc_metric):
        zc_weight = np.clip(zc_metric, 0.1, 1.0)

    # Empirically observed accuracy tables (Replace with your own calibration results)
    hoc_cal = np.array([[0, 0.45], [2.5, 0.55], [5, 0.68], [7.5, 0.78], [10, 0.86], [12.5, 0.91], [15, 0.94]])
    cyc_cal = np.array([[0, 0.42], [2.5, 0.58], [5, 0.71], [7.5, 0.82], [10, 0.88], [12.5, 0.92], [15, 0.95]])
    vit_cal = np.array([[0, 0.35], [2.5, 0.52], [5, 0.70], [7.5, 0.85], [10, 0.93], [12.5, 0.97], [15, 0.99]])

    def get_val(cal, s):
        f = interp1d(cal[:, 0], cal[:, 1], kind='linear', fill_value="extrapolate")
        return np.clip(f(s) * zc_weight, 0.05, 1.0)

    return {
        'hoc': get_val(hoc_cal, snr),
        'cyclo': get_val(cyc_cal, snr),
        'vit': get_val(vit_cal, snr)
    }

def calculate_bpa(probabilities, reliability, threshold):
    """
    Step 1 & 2: BPA Construction (Discounting).
    m(A) = r * P(A) | m(Theta) = 1 - r
    """
    bpa_core = reliability * probabilities
    total_belief = np.sum(bpa_core)
    uncertainty = max(1.0 - total_belief, threshold)
    
    if (total_belief + uncertainty) > 1.0:
        scale = (1.0 - uncertainty) / total_belief
        bpa_core *= scale
        
    return np.append(bpa_core, uncertainty)

def combine_evidence(bpa1, bpa2):
    """
    Step 3: Dempster's Rule of Combination.
    Normalizes by conflict factor K.
    """
    num_classes = len(FUSION_CLASSES)
    conflict_k = 0.0
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                conflict_k += bpa1[i] * bpa2[j]
    
    if conflict_k >= 1.0:
        return np.append(np.zeros(num_classes), 1.0), 1.0
        
    norm = 1.0 / (1.0 - conflict_k)
    combined = np.zeros(len(bpa1))
    
    for i in range(num_classes):
        mass = (bpa1[i] * bpa2[i]) + (bpa1[i] * bpa2[-1]) + (bpa1[-1] * bpa2[i])
        combined[i] = mass * norm
    combined[-1] = (bpa1[-1] * bpa2[-1]) * norm
    
    return combined, conflict_k

def resolve_conflict(bpas, rels):
    """Fallback to Weighted Average for high conflict cases."""
    total_rel = sum(rels.values())
    if total_rel <= 0: return np.append(np.zeros(len(FUSION_CLASSES)), 1.0)
    avg_bpa = (rels['hoc'] * bpas['hoc'] + rels['cyclo'] * bpas['cyclo'] + rels['vit'] * bpas['vit']) / total_rel
    return avg_bpa / np.sum(avg_bpa)

def extract_id(filename):
    """Helper to extract ID## from ViT filenames for sorting."""
    match = re.search(r'ID(\d+)', str(filename))
    return int(match.group(1)) if match else 0

def run_ds_fusion():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # 1. Load and Synchronize CSVs
    print("Loading and Synchronizing Classifiers...")
    df_vit = pd.read_csv(VIT_CSV_PATH)
    df_hoc = pd.read_csv(HOC_CSV_PATH)
    df_cyc = pd.read_csv(CYCLO_CSV_PATH)

    # Standardize ViT to have a 'FrameIndex' for merging
    if 'filename' in df_vit.columns:
        df_vit['FrameIndex'] = df_vit['filename'].apply(extract_id)
    
    # Ensure column casing is uniform
    for df in [df_vit, df_hoc, df_cyc]:
        df.columns = [c.lower() for c in df.columns]

    # Prob column mappings (Aligning index 0-4 to BPSK, QPSK, 8PSK, 16QAM, 64QAM)
    # ViT uses prob_qpsk, HOC/Cyclo use prob_4psk
    vit_prob_cols = ['prob_bpsk', 'prob_qpsk', 'prob_8psk', 'prob_16qam', 'prob_64qam']
    mat_prob_cols = ['prob_bpsk', 'prob_4psk', 'prob_8psk', 'prob_16qam', 'prob_64qam']

    # Rename ALL classifier columns with explicit suffixes BEFORE merge
    vit_rename = {c: c + '_vit' for c in df_vit.columns if c != 'frameindex'}
    hoc_rename = {c: c + '_hoc' for c in df_hoc.columns if c != 'frameindex'}
    cyc_rename = {c: c + '_cyc' for c in df_cyc.columns if c != 'frameindex'}
    
    df_vit = df_vit.rename(columns=vit_rename)
    df_hoc = df_hoc.rename(columns=hoc_rename)
    df_cyc = df_cyc.rename(columns=cyc_rename)

    # Merge on frameindex
    df_merged = df_vit.merge(df_hoc, on='frameindex')
    df_merged = df_merged.merge(df_cyc, on='frameindex')
    df_merged = df_merged.sort_values('frameindex').reset_index(drop=True)

    n_samples = len(df_merged)
    print(f"Aligned {n_samples} frames. Processing...")

    # Build actual column names with suffixes
    vit_cols_suffixed = [c + '_vit' for c in vit_prob_cols]
    hoc_cols_suffixed = [c + '_hoc' for c in mat_prob_cols]
    cyc_cols_suffixed = [c + '_cyc' for c in mat_prob_cols]

    p_vit = df_merged[vit_cols_suffixed].values
    p_hoc = df_merged[hoc_cols_suffixed].values
    p_cyc = df_merged[cyc_cols_suffixed].values

    y_true = df_merged['true_label_vit'].str.upper()
    label_map = {name.upper(): i for i, name in enumerate(FUSION_CLASSES)}
    label_map['4PSK'] = 1
    y_true_indices = y_true.map(label_map).values

    fused_preds, fused_confs, conflict_values = [], [], []

    for i in range(n_samples):
        snr = df_merged['snr_vit'].iloc[i]
        zc_val = df_merged['zc_peak'].iloc[i] if 'zc_peak' in df_merged.columns else None
        rels = calculate_reliability(snr, zc_metric=zc_val)
        
        bpa_v = calculate_bpa(p_vit[i], rels['vit'], UNCERTAINTY_THRESHOLD)
        bpa_h = calculate_bpa(p_hoc[i], rels['hoc'], UNCERTAINTY_THRESHOLD)
        bpa_c = calculate_bpa(p_cyc[i], rels['cyclo'], UNCERTAINTY_THRESHOLD)
        
        bpa_hc, k1 = combine_evidence(bpa_h, bpa_c)
        bpa_final, k2 = combine_evidence(bpa_hc, bpa_v)
        total_k = 1.0 - (1.0 - k1) * (1.0 - k2)
        
        if total_k > CONFLICT_THRESHOLD:
            bpa_final = resolve_conflict({'hoc': bpa_h, 'cyclo': bpa_c, 'vit': bpa_v}, rels)

        beliefs = bpa_final[:5]
        final_probs = beliefs / np.sum(beliefs) if np.sum(beliefs) > 0 else np.ones(5)/5
        
        best_idx = np.argmax(final_probs)
        fused_preds.append(best_idx)
        fused_confs.append(final_probs[best_idx])
        conflict_values.append(total_k)

    # Analysis
    df_merged['ds_prediction'] = [FUSION_CLASSES[idx] for idx in fused_preds]
    df_merged['ds_correct'] = (df_merged['ds_prediction'].str.upper() == y_true).astype(int)
    
    acc = df_merged['ds_correct'].mean() * 100
    print(f"\nAligned DS Fusion Accuracy: {acc:.2f}%")
    
    df_merged.to_csv(os.path.join(OUTPUT_DIR, 'DS_Fused_Results.csv'), index=False)
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true_indices, fused_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=FUSION_CLASSES, yticklabels=FUSION_CLASSES)
    plt.title(f'DS Fusion Confusion Matrix (Aligned)\nAccuracy: {acc:.2f}%')
    plt.savefig(os.path.join(OUTPUT_DIR, 'DS_Confusion_Matrix.png'))

if __name__ == "__main__":
    run_ds_fusion()