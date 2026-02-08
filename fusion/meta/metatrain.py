import numpy as np
import pandas as pd
import os
import sys
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- USER CONFIGURATION: Set the specific paths to your files here ---
VIT_NPY_PATH   = r'D:\w\Documents\199\data\Phase1_Wired_Dataset\vit_npy_data\vit_results.npy'
HOC_NPY_PATH   = r'D:\w\Documents\199\data\Phase1_Wired_Dataset\HOC_Classification_Results\HOC_Results.npy'
CYCLO_NPY_PATH = r'D:\w\Documents\199\data\Phase1_Wired_Dataset\Cyclo_Classification_Results\Cyclo_Results.npy'
VIT_CSV_PATH   = r'D:\w\Documents\199\data\Phase1_Wired_Dataset\vit_npy_data\vit_results.csv'

# Directory where the trained model and plots will be saved
OUTPUT_DIR     = r'D:\w\Documents\199\data'
# ---------------------------------------------------------------------

FUSION_CLASSES = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM']
MODEL_SAVE_NAME = 'Meta_Classifier_Model.pkl'

def load_consolidated_data():
    """
    Loads specific .npy and .csv files provided in the configuration.
    """
    # 1. Validation check
    paths = [VIT_NPY_PATH, HOC_NPY_PATH, CYCLO_NPY_PATH, VIT_CSV_PATH]
    for p in paths:
        if not os.path.exists(p):
            print(f"Error: File not found -> {p}")
            return None, None

    print("Loading probability matrices...")
    try:
        p_vit = np.load(VIT_NPY_PATH)
        p_hoc = np.load(HOC_NPY_PATH)
        p_cyc = np.load(CYCLO_NPY_PATH)
    except Exception as e:
        print(f"Error loading .npy files: {e}")
        return None, None
    
    # 2. Load ground truth labels from the ViT results CSV
    print(f"Loading ground truth from: {os.path.basename(VIT_CSV_PATH)}")
    df_labels = pd.read_csv(VIT_CSV_PATH)
    
    # Standardize column names for label extraction
    df_labels.columns = [c.lower() for c in df_labels.columns]
    label_col = 'true_label' if 'true_label' in df_labels.columns else 'truelabel'
    
    if label_col not in df_labels.columns:
        print(f"Error: Could not find ground truth column in {VIT_CSV_PATH}")
        return None, None

    # Map string labels to indices for the classifier
    label_map = {name.upper(): i for i, name in enumerate(FUSION_CLASSES)}
    label_map['4PSK'] = 1  # Handle MATLAB/Python naming differences
    
    y_labels = df_labels[label_col].str.upper().map(label_map).values

    # 3. Alignment check
    n_samples = min(len(p_vit), len(p_hoc), len(p_cyc), len(y_labels))
    if len(p_vit) != n_samples or len(y_labels) != n_samples:
        print(f"Warning: Sample count mismatch. Aligning to {n_samples} samples.")
    
    # Slice all matrices to the shortest length to ensure row-alignment
    p_vit = p_vit[:n_samples]
    p_hoc = p_hoc[:n_samples]
    p_cyc = p_cyc[:n_samples]
    y = y_labels[:n_samples]

    # Stack features horizontally: [ViT(5) | HOC(5) | Cyclo(5)] = 15 total features
    x = np.hstack([p_vit, p_hoc, p_cyc])
    
    print(f"Successfully aligned {n_samples} samples.")
    print(f"Meta-feature matrix shape: {x.shape}")
    
    return x, y

def train_and_evaluate(X, Y):
    print("\n" + "="*45)
    print("   TRAINING STACKED FUSION META-CLASSIFIER")
    print("="*45)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegression(solver='lbfgs', max_iter=2000)
    
    accuracies = []
    print(f"{'Fold':<6} | {'Accuracy':<10}")
    print("-" * 20)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, Y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"Fold {fold} | {acc*100:6.2f}%")
        
    avg_acc = np.mean(accuracies) * 100
    print("-" * 20)
    print(f"Mean Fusion Accuracy: {avg_acc:.2f}%\n")

    # Final training
    print("Training final model on full dataset...")
    model.fit(X, Y)
    
    # Save the model
    save_path = os.path.join(OUTPUT_DIR, MODEL_SAVE_NAME)
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {save_path}")

    # Visualization: Weights Heatmap
    plt.figure(figsize=(15, 8))
    feature_names = [f'ViT_{c}' for c in FUSION_CLASSES] + \
                    [f'HOC_{c}' for c in FUSION_CLASSES] + \
                    [f'Cyc_{c}' for c in FUSION_CLASSES]
    
    sns.heatmap(np.abs(model.coef_), annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=feature_names, yticklabels=FUSION_CLASSES)
    
    plt.title(f"Meta-Classifier Fusion Weights\nAverage Accuracy: {avg_acc:.2f}%")
    plt.xlabel("Input Probabilities from Base Classifiers")
    plt.ylabel("Target Output Class")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    weight_plot = os.path.join(OUTPUT_DIR, 'Fusion_Weights_Heatmap.png')
    plt.savefig(weight_plot)
    print(f"Weights heatmap saved to: {weight_plot}")

    # Visualization: Confusion Matrix
    y_final_pred = model.predict(X)
    cm = confusion_matrix(Y, y_final_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=FUSION_CLASSES, yticklabels=FUSION_CLASSES)
    plt.title('Fused System Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class (Meta-Fusion)')
    plt.tight_layout()
    
    cm_plot = os.path.join(OUTPUT_DIR, 'Fusion_Confusion_Matrix.png')
    plt.savefig(cm_plot)
    print(f"Confusion matrix saved to: {cm_plot}")
    
    print("\nFinal Classification Report:")
    print(classification_report(Y, y_final_pred, target_names=FUSION_CLASSES))

if __name__ == "__main__":
    X, Y = load_consolidated_data()
    
    if X is not None:
        train_and_evaluate(X, Y)