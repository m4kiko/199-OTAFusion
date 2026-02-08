import torch
import torch.nn.functional as F
import numpy as np
import os
import h5py
import yaml
import random
from runner.utils import get_config, model_selection
from data.dataset import FewShotDataset
from sklearn.metrics import confusion_matrix
import pandas as pd
import warnings
import sys  # --- ADDED: For command-line arguments ---

# --- Configuration ---
# Set this to False if you do not have an NVIDIA GPU or CUDA installed
USE_GPU = True
# --- End Configuration ---

# Suppress PyTorch warnings about UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

class MetaBlackBoxClassifier:
    def __init__(self, config_path='./config/config.yaml', model_params_path='./config/model_params.yaml'):
        """
        Initializes the classifier, loads the model, and builds the
        24-way reference support set.
        """
        print("Initializing Meta-Transformer black box classifier...")
        self.config = get_config(config_path)
        
        # --- FIX: Force the config to use the correct meta-transformer model ---
        self.config['model'] = 'vit_main'
        # --- End Fix ---
        
        self.model_params = get_config(model_params_path)[self.config['model']]
        
        # Check for CUDA
        if USE_GPU and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.config['gpu_ids'][0]}")
            print(f"Running on GPU: {self.device}")
        else:
            self.device = torch.device("cpu")
            self.config['cuda'] = False
            print("Running on CPU")

        # Load the Meta-Transformer model (vit_main)
        self.model = model_selection(self.config, self.model_params, mode='test')
        
        # Load pre-trained weights
        model_path = os.path.join(self.config['load_test_path'], self.config['model'], self.config['load_model_name'])
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print(f"ERROR: Model file not found at {model_path}")
            print("Please follow the README to download and place the pre-trained models.")
            raise
            
        self.model.to(self.device)
        self.model.eval()
        
        # Get all 24 class names from the config
        self.total_class_names = self.config['total_class']
        
        # Check for mismatch
        # --- FIX: Access the final layer correctly for ProtoNet(ViT) ---
        model_output_size = self.model.encoder.fc.in_features
        expected_size = self.model_params.get('embed_dim')
        
        if model_output_size != expected_size:
            print(f"Warning: Model embed_dim ({expected_size}) does not match encoder output ({model_output_size}).")
        # --- End Fix ---

        # --- Build and cache the 24-way reference support set ---
        print("Building 24-way reference support set (prototypes)...")
        self.reference_prototypes, self.reference_class_labels = self._build_reference_support_set()
        
        print("Classifier ready.")

    def _preprocess_signal(self, signal_array_2_1024):
        """
        Prepares a single raw signal array for classification.
        Input: (2, 1024) numpy array
        Output: (1, 1, 2, 1024) torch tensor
        """
        # (2, 1024) -> (1, 2, 1024)
        signal_tensor = torch.from_numpy(signal_array_2_1024).float()
        # (1, 2, 1024) -> (1, 1, 2, 1024)
        signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0)
        return signal_tensor.to(self.device)

    def _build_reference_support_set(self):
        """
        Loads a 24-way, 5-shot support set from the HDF5 file
        and pre-computes the prototypes.
        """
        # 1. Load a 24-class episode
        demo_config = self.config.copy()
        all_indices = list(range(len(demo_config['total_class'])))
        demo_config['test_class_indices'] = all_indices

        try:
            test_data = FewShotDataset(demo_config,
                                       mode='test', 
                                       snr_range=[20, 20], # Use high SNR for clean prototypes
                                       sample_len=demo_config["train_sample_len"],
                                       train_sample_len=demo_config["train_sample_len"])
        except FileNotFoundError:
            print(f"ERROR: Could not load HDF5 dataset to build support set.")
            print(f"Please check the path in config.yaml: {demo_config['dataset_path']}")
            raise

        if len(test_data) == 0:
            print("Error: Could not load FewShotDataset. Is the dataset file present?")
            raise Exception("Dataset loading failed")

        episode_data = test_data[0] # Get the first available episode
        class_indices = list(episode_data.keys())
        class_labels_present = [self.total_class_names[i] for i in class_indices]

        # 2. Build the support_set_dict
        support_set_dict = {}
        for idx in class_indices:
            support_set_dict[idx] = episode_data[idx]['support']
        
        # 3. Pre-process and compute prototypes
        n_way = len(class_indices)
        n_support = self.config['num_support']
        
        x_support = []
        for label_idx in class_indices:
            support_signals = support_set_dict[label_idx]
            # Process each signal in the support set
            # --- FIX: Ensure signals are (1, 2, 1024) ---
            processed_signals = [torch.from_numpy(sig).float().unsqueeze(0) for sig in support_signals]
            x_support.extend(processed_signals)
        
        # Stack all support signals into a batch
        # --- FIX: Add batch dim [bs, 1, 2, 1024] ---
        x_support_tensor = torch.stack(x_support).to(self.device) 
        
        # Get embeddings from the encoder
        with torch.no_grad():
            z_support = self.model.encoder.forward(x_support_tensor)
        
        z_support_dim = z_support.size(-1)
        
        # Average embeddings to create prototypes
        # Shape: (n_way, n_support, z_dim) -> (n_way, z_dim)
        prototypes_tensor = z_support.view(n_way, n_support, z_support_dim).mean(1)
        
        return prototypes_tensor, class_labels_present

    def classify_signal(self, query_signal, prototypes_tensor, class_labels):
        """
        Classifies a single query signal against a pre-computed
        set of class prototypes.
        """
        
        # 1. Pre-process the query signal
        # Input shape is (2, 1024), model expects (batch, 1, 2, 1024)
        x_query = self._preprocess_signal(query_signal)

        # 2. Get query signal embedding
        with torch.no_grad():
            z_query = self.model.encoder.forward(x_query) # Shape: (1, z_dim)

        # 3. Compute distances to all class prototypes
        # z_query: (1, z_dim)
        # prototypes_tensor: (n_way, z_dim)
        # cdist output shape: (1, n_way)
        dists = torch.cdist(z_query, prototypes_tensor)

        # 4. Convert distances to probabilities
        # Lower distance = higher probability
        # We use softmax on the *negative* distances
        log_p_y = F.log_softmax(-dists, dim=1)
        probabilities = torch.exp(log_p_y).cpu().numpy()[0] # Get the first (only) item in batch

        # 5. Get the best guess
        prediction_index = np.argmax(probabilities)
        guess = class_labels[prediction_index]
        probability = probabilities[prediction_index]
        
        # 6. Create the full probability dictionary
        all_probabilities = {label: prob for label, prob in zip(class_labels, probabilities)}
        
        return guess, probability, all_probabilities

    def classify_from_file(self, npy_file_path):
        """
        High-level wrapper to load a .npy file and classify it
        against the cached 24-way reference set.
        """
        # Load the signal file
        try:
            signal_array = np.load(npy_file_path)
        except FileNotFoundError:
            raise
        except Exception as e:
            raise ValueError(f"Could not read .npy file. Error: {e}")
            
        # Validate shape
        if signal_array.shape != (2, 1024):
            raise ValueError(f"Invalid signal shape. Expected (2, 1024), but got {signal_array.shape}. Please reshape your .npy file.")
            
        # Classify using the cached reference set
        guess, prob, all_probs = self.classify_signal(
            signal_array,
            self.reference_prototypes,
            self.reference_class_labels
        )
        
        return guess, prob, all_probs

# --- Full Evaluation Function (from previous step) ---
def run_full_evaluation(classifier):
    """
    Loads a full 24-class test episode and evaluates all query signals
    against a single 24-way support set.
    """
    print("\n--- Starting Full 24-Way Evaluation ---")
    config = classifier.config

    # --- Evaluation Parameters ---
    TARGET_SNR = 20  # SNR to test at
    NUM_SUPPORT_SHOTS = config['num_support'] # From config.yaml, typically 5
    NUM_QUERY_SHOTS = config['num_query']     # From config.yaml, typically 10
    # --- End Parameters ---
    
    print(f"SNR: {TARGET_SNR}dB")
    print(f"Shots (k): {NUM_SUPPORT_SHOTS} support, {NUM_QUERY_SHOTS} query per class")
    
    # 1. Load a full 24-class episode
    print("Loading evaluation data (all 24 classes)...")
    eval_config = config.copy()
    all_indices = list(range(len(eval_config['total_class'])))
    eval_config['test_class_indices'] = all_indices

    try:
        test_data = FewShotDataset(eval_config,
                                   mode='test', 
                                   snr_range=[TARGET_SNR, TARGET_SNR],
                                   sample_len=eval_config["train_sample_len"],
                                   train_sample_len=eval_config["train_sample_len"])
    except FileNotFoundError:
        print(f"ERROR: Could not load HDF5 dataset for evaluation.")
        raise
    except Exception as e:
        print(f"ERROR: Failed to load FewShotDataset. {e}")
        return

    if len(test_data) == 0:
        print("Error: Could not load evaluation data. Is the dataset file present?")
        return

    # Use the classifier's pre-built reference set
    prototypes_tensor = classifier.reference_prototypes
    class_labels = classifier.reference_class_labels
    class_indices = [classifier.total_class_names.index(label) for label in class_labels]

    print("Classifying 240 total query signals...")
    
    all_query_signals = {}
    all_actual_labels = []
    all_predicted_labels = []
    
    # We need to get the query signals from the dataset
    # The reference set was built from episode 0, so we use episode 1+ for queries
    # To be robust, let's pull query signals from multiple episodes
    
    query_signals_by_class = {label: [] for label in class_labels}
    
    # Gather enough query signals
    for i in range(len(test_data)):
        episode_data = test_data[i]
        for class_idx in class_indices:
            class_name = classifier.total_class_names[class_idx]
            if class_name in class_labels: # Ensure class is in our set
                query_signals = episode_data[class_idx]['query']
                query_signals_by_class[class_name].extend(query_signals)
    
    total_classified = 0
    total_correct = 0
    
    # This will store (actual_label, predicted_label) for the confusion matrix
    cm_pairs = []

    for class_name in class_labels:
        # Get the first N query signals for this class
        signals_to_test = query_signals_by_class[class_name][:NUM_QUERY_SHOTS]
        
        class_correct = 0
        for query_signal in signals_to_test:
            # Classify the signal
            guess, prob, _ = classifier.classify_signal(
                query_signal,
                prototypes_tensor,
                class_labels
            )
            
            cm_pairs.append((class_name, guess))
            
            if guess == class_name:
                class_correct += 1
                
        total_classified += len(signals_to_test)
        total_correct += class_correct
        print(f"  - Tested {class_name:<10}: {class_correct}/{len(signals_to_test)} correct")


    # --- Print Final Report ---
    print("\n\n--- Full Evaluation Results ---")
    overall_accuracy = (total_correct / total_classified) * 100
    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_classified})")

    # Per-class accuracy
    print("\n--- Accuracy Per Modulation Type ---")
    per_class_acc = {}
    for class_name in class_labels:
        correct = sum(1 for actual, pred in cm_pairs if actual == class_name and pred == class_name)
        total = sum(1 for actual, _ in cm_pairs if actual == class_name)
        acc = (correct / total) * 100 if total > 0 else 0
        per_class_acc[class_name] = acc
    
    # Sort by accuracy (descending)
    sorted_per_class_acc = sorted(per_class_acc.items(), key=lambda item: item[1], reverse=True)
    for class_name, acc in sorted_per_class_acc:
        print(f"{class_name:<10}: {acc:.2f}%")

    # Confusion Matrix
    print("\n--- Confusion Matrix (Actual vs. Predicted) ---")
    actuals = [pair[0] for pair in cm_pairs]
    preds = [pair[1] for pair in cm_pairs]
    
    cm = confusion_matrix(actuals, preds, labels=class_labels)
    
    # Use pandas for a clean print
    df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    
    # Find columns that have at least one non-zero value to keep the table readable
    relevant_cols = [col for col in class_labels if df[col].sum() > 0]
    print(df.loc[:, relevant_cols])


def create_demo_npy_file_if_not_exists(filename="my_signal.npy"):
    """
    Pulls one signal from the HDF5 file and saves it as 'my_signal.npy'
    if it doesn't already exist. This is just for demonstration.
    """
    if os.path.exists(filename):
        return
        
    print(f"\nDemo file '{filename}' not found. Creating one from the dataset...")
    
    # Load HDF5 file to grab one signal
    try:
        config = get_config('./config/config.yaml')
        data_path = os.path.join(config['dataset_path'], "GOLD_XYZ_OSC.0001_1024.hdf5")
        data = h5py.File(data_path, 'r')
    except Exception as e:
        print(f"ERROR: Could not open HDF5 dataset at {data_path} to create demo file.")
        print("Please ensure the dataset is downloaded and paths are correct.")
        return

    # Find the first signal with 20dB SNR and class QPSK (index 4)
    snrs = data['Z'][:]
    labels = data['Y'][:]
    
    target_snr = 20
    target_label_idx = 4 # QPSK
    
    demo_signal_idx = -1
    for i in range(len(snrs)):
        if snrs[i] == target_snr and labels[i][target_label_idx] == 1:
            demo_signal_idx = i
            break
            
    if demo_signal_idx == -1:
        print("Could not find a suitable demo signal in the dataset.")
        return

    # Get the signal and format it as (2, 1024)
    # Original shape is (1024, 2), so we transpose it
    demo_signal = data['X'][demo_signal_idx].transpose() 
    
    # Save it as a .npy file
    np.save(filename, demo_signal.astype(np.float32))
    print(f"Saved demo QPSK signal as '{filename}'")
    data.close()


if __name__ == "__main__":
    
    RUN_FULL_EVALUATION = False # Set to False to classify a single file
    
    # --- MODIFIED: Accept signal file path from command line ---
    if len(sys.argv) > 1:
        YOUR_SIGNAL_FILE = sys.argv[1]
    else:
        YOUR_SIGNAL_FILE = '8PSK_snr_22_len_1024_idx_619308.npy'
    # --- END MODIFICATION ---
    
    try:
        # 2. Initialize the classifier (this builds the reference set)
        classifier = MetaBlackBoxClassifier()
        
        if RUN_FULL_EVALUATION:
            run_full_evaluation(classifier)
        else:
            # 3. Create a demo file if the target file doesn't exist
            create_demo_npy_file_if_not_exists(YOUR_SIGNAL_FILE)
        
            print(f"\n--- Classifying External Signal ---")
            print(f"Loading signal from: {YOUR_SIGNAL_FILE}")
        
            # 4. Classify your file
            guess, prob, all_probs = classifier.classify_from_file(YOUR_SIGNAL_FILE)
            
            # 5. Print results
            print("\n--- Classification Result ---")
            print(f"Predicted Class: {guess}")
            print(f"Confidence:      {prob * 100:.2f}%")

            # Optional: Print top 5 probabilities
            print("\n--- Top 5 Probabilities ---")
            sorted_probs = sorted(all_probs.items(), key=lambda item: item[1], reverse=True)
            for mod, p in sorted_probs[:5]:
                print(f"{mod:<10}: {p*100:.2f}%")

    except FileNotFoundError as e:
        print(f"\n--- FATAL ERROR ---")
        print(f"File not found: {e}")
        print("Please ensure the HDF5 dataset and pre-trained models are downloaded and in the correct folders.")
    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED ---")
        print(e)