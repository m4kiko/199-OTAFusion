import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.interpolate import interp1d

# --- USER CONFIGURATION: Point to the Consolidated CSV ---
CONSOLIDATED_CSV_PATH = r'D:\w\Documents\199\data\Phase1_Wired_Dataset\Consolidated\Consolidated_AMC_Results.csv'

# Output location
OUTPUT_DIR = r'D:\w\Documents\199\data\Phase1_Wired_Dataset\DS_Fusion_Results'

# DS Parameters
UNCERTAINTY_THRESHOLD = 0.05
CONFLICT_THRESHOLD = 0.95
FUSION_CLASSES = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM']

# OTA Diagnostic Flags
ENABLE_DIAGNOSTIC_PLOTS = True
# ---------------------------------------------------------

def calculate_reliability(snr, zc_metric=None):
    """
    Step 0: Reliability Calibration (Context-Aware Weighting).
    Uses SNR-based accuracy lookup and optional ZC peak scaling.
    """
    zc_weight = 1.0
    if zc_metric is not None and not pd.isna(zc_metric):
        zc_weight = np.clip(zc_metric, 0.1, 1.0)

    # Calibration Tables (Based on Empirical Baseline)
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
    """Fallback Strategy: Weighted Average."""
    total_rel = sum(rels.values())
    if total_rel <= 0: return np.append(np.zeros(len(FUSION_CLASSES)), 1.0)
    avg_bpa = (rels['hoc'] * bpas['hoc'] + rels['cyclo'] * bpas['cyclo'] + rels['vit'] * bpas['vit']) / total_rel
    return avg_bpa / np.sum(avg_bpa)

def plot_mass_distribution(bpa, frame_idx, snr, conflict):
    """Diagnostic Plot: Mass Distribution for a single frame."""
    plt.figure(figsize=(8, 4))
    labels = FUSION_CLASSES + ['Uncertainty (Θ)']
    plt.bar(labels, bpa, color=['#3498db']*5 + ['#e74c3c'])
    plt.title(f'Frame {frame_idx} (SNR {snr}dB) | Conflict K: {conflict:.2f}')
    plt.ylabel('Belief Mass')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

def run_ds_fusion():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    print("Loading Consolidated Dataset...")
    if not os.path.exists(CONSOLIDATED_CSV_PATH):
        print(f"Error: {CONSOLIDATED_CSV_PATH} not found. Run consolidate_results.py first.")
        return

    df = pd.read_csv(CONSOLIDATED_CSV_PATH)
    n_samples = len(df)
    print(f"Processing {n_samples} aligned frames...")

    # Define probability columns as established by the consolidation script
    vit_cols = ['prob_bpsk_vit', 'prob_qpsk_vit', 'prob_8psk_vit', 'prob_16qam_vit', 'prob_64qam_vit']
    hoc_cols = ['prob_bpsk_hoc', 'prob_qpsk_hoc', 'prob_8psk_hoc', 'prob_16qam_hoc', 'prob_64qam_hoc']
    cyc_cols = ['prob_bpsk_cyc', 'prob_qpsk_cyc', 'prob_8psk_cyc', 'prob_16qam_cyc', 'prob_64qam_cyc']

    # --- LABEL NORMALIZATION ---
    # Ensure truelabel matches FUSION_CLASSES exactly (Fixes the 0.00% accuracy issue)
    df['truelabel'] = df['truelabel'].str.upper().replace('4PSK', 'QPSK')
    y_true = df['truelabel']
    
    label_map = {name: i for i, name in enumerate(FUSION_CLASSES)}
    y_true_indices = y_true.map(label_map).values

    fused_preds = []
    fused_confs = []
    conflict_values = []

    for i in range(n_samples):
        snr = df['snr'].iloc[i]
        zc_val = df['zc_peak'].iloc[i] if 'zc_peak' in df.columns else None
        rels = calculate_reliability(snr, zc_metric=zc_val)
        
        # BPA Construction
        bpa_v = calculate_bpa(df[vit_cols].iloc[i].values, rels['vit'], UNCERTAINTY_THRESHOLD)
        bpa_h = calculate_bpa(df[hoc_cols].iloc[i].values, rels['hoc'], UNCERTAINTY_THRESHOLD)
        bpa_c = calculate_bpa(df[cyc_cols].iloc[i].values, rels['cyclo'], UNCERTAINTY_THRESHOLD)
        
        # Rule of Combination
        bpa_hc, k1 = combine_evidence(bpa_h, bpa_c)
        bpa_final, k2 = combine_evidence(bpa_hc, bpa_v)
        total_k = 1.0 - (1.0 - k1) * (1.0 - k2)
        
        # Conflict Handling
        if total_k > CONFLICT_THRESHOLD:
            bpa_final = resolve_conflict({'hoc': bpa_h, 'cyclo': bpa_c, 'vit': bpa_v}, rels)

        # Decision Rule (Maximum Mass Criterion)
        beliefs = bpa_final[:5]
        final_probs = beliefs / np.sum(beliefs) if np.sum(beliefs) > 0 else np.ones(5)/5
        
        best_idx = np.argmax(final_probs)
        fused_preds.append(best_idx)
        fused_confs.append(final_probs[best_idx])
        conflict_values.append(total_k)

        # Periodic Diagnostic Plot
        if ENABLE_DIAGNOSTIC_PLOTS and i % 1000 == 0:
            plot_mass_distribution(bpa_final, df['frameindex'].iloc[i], snr, total_k)

    # Export Results
    df['ds_prediction'] = [FUSION_CLASSES[idx] for idx in fused_preds]
    df['ds_confidence'] = fused_confs
    df['ds_conflict'] = conflict_values
    df['ds_correct'] = (df['ds_prediction'].str.upper() == y_true.str.upper()).astype(int)
    
    acc = df['ds_correct'].mean() * 100
    print(f"\nFinal DS Fusion Accuracy (Consolidated): {acc:.2f}%")
    
    # --- DEBUGGING OUTPUT ---
    if acc == 0 and n_samples > 0:
        print("\nDEBUG: Accuracy is 0.00%. Checking for label mismatch...")
        print(f"Sample True Label: '{y_true.iloc[0]}'")
        print(f"Sample Predicted Label: '{df['ds_prediction'].iloc[0]}'")
        print(f"Available FUSION_CLASSES: {FUSION_CLASSES}")

    output_path = os.path.join(OUTPUT_DIR, 'DS_Fused_Results.csv')
    df.to_csv(output_path, index=False)
    
    # Global Performance Plots
    plt.figure(figsize=(10, 8))
    # Drop NaNs if mapping failed to prevent plot errors
    valid_mask = ~np.isnan(y_true_indices)
    if np.any(valid_mask):
        cm = confusion_matrix(y_true_indices[valid_mask], np.array(fused_preds)[valid_mask])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=FUSION_CLASSES, yticklabels=FUSION_CLASSES)
        plt.title(f'Consolidated DS Fusion Confusion Matrix\nAccuracy: {acc:.2f}%')
        plt.savefig(os.path.join(OUTPUT_DIR, 'DS_Confusion_Matrix.png'))
        print(f"Results and confusion matrix saved to: {OUTPUT_DIR}")
    else:
        print("Warning: No valid labels found for Confusion Matrix.")

if __name__ == "__main__":
    run_ds_fusion()