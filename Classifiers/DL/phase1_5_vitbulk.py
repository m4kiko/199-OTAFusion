import numpy as np
import sys
import os
import pandas as pd
# Import the reporting function from the Canvas
from report import generate_readable_report

# Import the blackbox code (Ensure blackbox_model.py is in the same directory)
try:
    from blackbox_model import MetaBlackBoxClassifier
except ImportError:
    print("Error: Could not find 'blackbox_model.py'. Ensure the ViT blackbox code is present.")
    sys.exit(1)

class CooperativeViTWrapper:
    def __init__(self):
        # Initialize the underlying 24-way classifier
        self.base_classifier = MetaBlackBoxClassifier()
        
        # Define the specific 5-class target subset for your project
        self.target_classes = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM']

    def classify_and_renormalize(self, npy_file_path):
        """
        Classifies the signal and filters output to only the 5 target classes.
        Renormalizes probabilities so they sum to 1.
        """
        try:
            # 1. Get raw 24-way probabilities from blackbox
            guess_24, prob_24, all_probs_24 = self.base_classifier.classify_from_file(npy_file_path)

            # 2. Extract only the 5 target classes
            target_probs = {}
            for cls in self.target_classes:
                target_probs[cls] = all_probs_24.get(cls, 0.0)

            # 3. Calculate normalization constant (Sum of subset)
            subset_sum = sum(target_probs.values())

            # 4. Renormalize to ensure Total Probability = 1.0
            if subset_sum > 0:
                final_probs = {cls: p / subset_sum for cls, p in target_probs.items()}
            else:
                # Fallback if no probability was assigned to the subset
                final_probs = {cls: 1.0/len(self.target_classes) for cls in self.target_classes}

            # 5. Determine new top guess within the subset
            best_mod = max(final_probs, key=final_probs.get)
            best_conf = final_probs[best_mod]

            return best_mod, best_conf, final_probs
        except Exception as e:
            print(f"Error processing {npy_file_path}: {e}")
            return None, None, None

    def _process_single_folder(self, folder_path, label=None):
        """Helper to process all .npy files in one folder."""
        results = []
        files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            guess, confidence, probs = self.classify_and_renormalize(file_path)
            
            if guess:
                # Extract SNR from filename if present (MOD_SNR##_ID##)
                snr = None
                try:
                    if "_SNR" in filename:
                        snr = float(filename.split('_SNR')[1].split('_')[0])
                except:
                    pass

                row = {
                    'filename': filename,
                    'true_label': label if label else "Unknown",
                    'prediction': guess,
                    'confidence': confidence,
                    'snr': snr
                }
                for mod, p in probs.items():
                    row[f'prob_{mod}'] = p
                results.append(row)
        return results

    def run(self, input_path):
        """Main entry point for Single Shot, Bulk (Mod), or Whole Folder."""
        # 1. Single Shot Mode
        if os.path.isfile(input_path):
            print(f"Mode: Single Shot - {input_path}")
            guess, confidence, probs = self.classify_and_renormalize(input_path)
            if guess:
                print(f"Prediction: {guess} ({confidence*100:.2f}%)")
                output_dir = os.path.dirname(input_path)
                output_csv = os.path.join(output_dir, "single_result.csv")
                df = pd.DataFrame([{
                    'filename': os.path.basename(input_path),
                    'prediction': guess,
                    'confidence': confidence,
                    **{f'prob_{k}': v for k, v in probs.items()}
                }])
                df.to_csv(output_csv, index=False)
                
                # Call report generator for single result
                print("Generating readable report...")
                generate_readable_report(output_csv)
            return

        # Check if it is a directory
        if not os.path.isdir(input_path):
            print(f"Error: {input_path} is not a valid file or directory.")
            return

        # Check for subdirectories to distinguish between Mode 2 and Mode 3
        subdirs = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]

        all_results = []
        output_csv = os.path.join(input_path, "vit_results.csv")

        if len(subdirs) > 0:
            # Mode 3: Classify Whole Folder (Iterate through subfolders)
            print(f"Mode: Whole Folder Scan - {input_path}")
            for sd in subdirs:
                sd_path = os.path.join(input_path, sd)
                print(f"  Processing Modulation Class: {sd}")
                all_results.extend(self._process_single_folder(sd_path, label=sd))
        else:
            # Mode 2: Bulk Classify by Modulation Class (Current folder is the mod)
            mod_label = os.path.basename(input_path)
            print(f"Mode: Bulk Modulation Class - {mod_label}")
            all_results.extend(self._process_single_folder(input_path, label=mod_label))

        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(output_csv, index=False)
            print(f"\nProcessing complete. Results saved to: {output_csv}")
            
            # --- CALL THE FUNCTION FROM CANVAS ---
            print("Generating readable reports and visualization plots...")
            generate_readable_report(output_csv)
        else:
            print("No .npy files found to process.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  1. Single file:   python vit_inference_wrapper.py path/to/signal.npy")
        print("  2. Mod folder:    python vit_inference_wrapper.py path/to/BPSK_folder")
        print("  3. Whole folder:  python vit_inference_wrapper.py path/to/vit_npy_data")
        sys.exit(1)

    input_path = sys.argv[1].rstrip(os.sep)
    wrapper = CooperativeViTWrapper()
    wrapper.run(input_path)