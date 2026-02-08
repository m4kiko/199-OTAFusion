import numpy as np
import pandas as pd
import os
import sys
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- USER CONFIGURATION: Set your paths here ---
MODEL_PATH     = r'D:\w\Documents\199\data\Meta_Classifier_Model.pkl'
VIT_NPY_PATH   = r'D:\w\Documents\199\data\Phase1_Wired_Dataset\vit_npy_data\vit_results.npy'
HOC_NPY_PATH   = r'D:\w\Documents\199\data\Phase1_Wired_Dataset\HOC_Classification_Results\HOC_Results.npy'
CYCLO_NPY_PATH = r'D:\w\Documents\199\data\Phase1_Wired_Dataset\Cyclo_Classification_Results\Cyclo_Results.npy'
VIT_CSV_PATH   = r'D:\w\Documents\199\data\Phase1_Wired_Dataset\vit_npy_data\vit_results.csv'

# Output location for the fused results
OUTPUT_DIR     = r'D:\w\Documents\199\data\Phase1_Wired_Dataset\Fusion_Results'
# -----------------------------------------------

FUSION_CLASSES = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM']

def run_meta_inference():
    """
    Loads the trained Meta-Classifier and performs fusion on bulk data.
    """
    # 1. Validation and Setup
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    paths = [MODEL_PATH, VIT_NPY_PATH, HOC_NPY_PATH, CYCLO_NPY_PATH, VIT_CSV_PATH]
    for p in paths:
        if not os.path.exists(p):
            print(f"Error: Required file not found -> {p}")
            return

    # 2. Load Model
    print(f"Loading Meta-Classifier model: {os.path.basename(MODEL_PATH)}")
    with open(MODEL_PATH, 'rb') as f:
        meta_model = pickle.load(f)

    # 3. Load and Align Base Probabilities
    print("Loading base classifier outputs...")
    p_vit = np.load(VIT_NPY_PATH)
    p_hoc = np.load(HOC_NPY_PATH)
    p_cyc = np.load(CYCLO_NPY_PATH)
    
    # Load labels and metadata from original CSV
    df_meta = pd.read_csv(VIT_CSV_PATH)
    
    # Align rows
    n_samples = min(len(p_vit), len(p_hoc), len(p_cyc), len(df_meta))
    if n_samples != len(df_meta):
        print(f"Warning: Aligning data to {n_samples} samples.")
        df_meta = df_meta.iloc[:n_samples].copy()
    
    # Standardize Ground Truth Labels (Numeric for metrics)
    df_meta.columns = [c.lower() for c in df_meta.columns]
    label_col = 'true_label' if 'true_label' in df_meta.columns else 'truelabel'
    
    label_map = {name.upper(): i for i, name in enumerate(FUSION_CLASSES)}
    label_map['4PSK'] = 1
    y_true = df_meta[label_col].str.upper().map(label_map).values

    # 4. Perform Fusion (Concatenate to 15 features)
    # X_meta structure: [ViT_probs, HOC_probs, Cyc_probs]
    X_meta = np.hstack([p_vit[:n_samples], p_hoc[:n_samples], p_cyc[:n_samples]])
    
    print("Executing Fused Inference...")
    y_fused_pred_idx = meta_model.predict(X_meta)
    y_fused_conf = np.max(meta_model.predict_proba(X_meta), axis=1)

    # 5. Save Fused Results to CSV
    df_meta['fused_prediction'] = [FUSION_CLASSES[idx] for idx in y_fused_pred_idx]
    df_meta['fused_confidence'] = y_fused_conf
    df_meta['fused_correct'] = (df_meta['fused_prediction'].str.upper() == df_meta[label_col].str.upper()).astype(int)

    output_csv = os.path.join(OUTPUT_DIR, 'Final_Fused_Results.csv')
    df_meta.to_csv(output_csv, index=False)
    
    # 6. Performance Summary
    acc = accuracy_score(y_true, y_fused_pred_idx) * 100
    print("\n" + "="*40)
    print("       FINAL FUSION PERFORMANCE")
    print("="*40)
    print(f"Overall Fused Accuracy: {acc:.2f}%")
    print(f"Results saved to: {output_csv}")
    print("="*40)

    # 7. Generate Fused Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_fused_pred_idx)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=FUSION_CLASSES, yticklabels=FUSION_CLASSES)
    plt.title(f'Final Fused System Confusion Matrix\nAccuracy: {acc:.2f}%')
    plt.ylabel('True Modulation')
    plt.xlabel('Fused Prediction')
    plt.tight_layout()
    
    cm_path = os.path.join(OUTPUT_DIR, 'Fused_Confusion_Matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion Matrix saved: {cm_path}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_fused_pred_idx, target_names=FUSION_CLASSES))

if __name__ == "__main__":
    run_meta_inference()