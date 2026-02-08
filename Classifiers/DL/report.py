import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_readable_report(input_csv="vit_results.csv", output_txt="classification_report.txt", plot_file="snr_accuracy_plot.png", distance_plot="distance_simulation_plot.png"):
    """
    Transforms the raw CSV data into a human-readable summary report,
    generates an SNR vs Accuracy plot, and simulates distance-based performance.
    All outputs are saved in the same directory as the input_csv.
    """
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found. Run the inference wrapper first.")
        return

    # Determine the directory of the input CSV to save outputs in the same location
    input_dir = os.path.dirname(input_csv)
    
    # Define full paths for output files to ensure they are saved in the input directory
    output_txt_path = os.path.join(input_dir, output_txt)
    plot_file_path = os.path.join(input_dir, plot_file)
    distance_plot_path = os.path.join(input_dir, distance_plot)

    # Load data
    df = pd.read_csv(input_csv)
    
    # Extract metadata from filename (assuming format: MOD_SNR##_ID##.npy)
    def get_snr(filename):
        try:
            return float(filename.split('_SNR')[1].split('_')[0])
        except:
            return None

    def get_true_mod(filename):
        try:
            return filename.split('_SNR')[0]
        except:
            return "Unknown"

    df['snr'] = df['filename'].apply(get_snr)
    df['true_modulation'] = df['filename'].apply(get_true_mod)
    
    # Calculate correctness for accuracy plots
    df['is_correct'] = (df['prediction'] == df['true_modulation']).astype(int)
    
    # Filter out rows where SNR or true_modulation couldn't be determined
    df = df.dropna(subset=['snr'])
    df = df[df['true_modulation'] != "Unknown"]

    if df.empty:
        print("Warning: No valid SNR or Modulation metadata found in filenames. Skipping plots.")
    else:
        # --- PLOT 1: SNR vs Accuracy ---
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        accuracy_data = df.groupby(['true_modulation', 'snr'])['is_correct'].mean().reset_index()
        accuracy_data['accuracy'] = accuracy_data['is_correct'] * 100
        
        sns.lineplot(
            data=accuracy_data, 
            x='snr', 
            y='accuracy', 
            hue='true_modulation', 
            marker='o',
            linewidth=2.5
        )
        
        plt.title('ViT Classification: SNR vs. Accuracy per Modulation', fontsize=14)
        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.ylim(-5, 105)
        plt.legend(title='Modulation', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(plot_file_path)
        plt.close()
        print(f"Accuracy plot generated: {plot_file_path}")

        # --- PLOT 2: Distance Simulation (Friis Equation Mapping) ---
        # Methodology: 6dB loss = 2x distance. 
        # We assume SNR_max (15dB) is 'Reference Distance D0' (e.g., 1 meter).
        # Formula: D = D0 * 10^((SNR_max - SNR_current) / 20)
        reference_snr = df['snr'].max()
        accuracy_data['virtual_distance'] = 1 * (10 ** ((reference_snr - accuracy_data['snr']) / 20))
        
        plt.figure(figsize=(10, 6))
        sns.set_style("white")
        
        sns.lineplot(
            data=accuracy_data, 
            x='virtual_distance', 
            y='accuracy', 
            hue='true_modulation', 
            marker='s',
            linewidth=2.5
        )
        
        plt.title('Distance Simulation: Accuracy vs. Relative Distance (Friis Model)', fontsize=14)
        plt.xlabel('Relative Distance (D/D0)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.ylim(-5, 105)
        plt.legend(title='Modulation', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(distance_plot_path)
        plt.close()
        print(f"Distance simulation plot generated: {distance_plot_path}")

    with open(output_txt_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("COOPERATIVE AMC: VI-TRANSFORMER CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")

        # 1. Global Summary
        f.write("--- GLOBAL SUMMARY ---\n")
        total_samples = len(df)
        f.write(f"Total Samples Processed: {total_samples}\n")
        
        pred_counts = df['prediction'].value_counts()
        f.write("\nDistribution of Predicted Classes:\n")
        for mod, count in pred_counts.items():
            percentage = (count / total_samples) * 100
            f.write(f"  {mod:<10}: {count:>4} samples ({percentage:>5.1f}%)\n")
        
        avg_conf = df['confidence'].mean() * 100
        f.write(f"\nAverage System Confidence: {avg_conf:.2f}%\n\n")

        # 2. Per-SNR & Distance Mapping
        if not df.empty:
            f.write("--- DISTANCE SIMULATION (FRIIS MAPPING) ---\n")
            f.write(f"{'SNR (dB)':<10} | {'Rel. Distance':<15} | {'Avg. Accuracy':<15}\n")
            f.write("-" * 45 + "\n")
            
            snr_groups = df.groupby('snr')['is_correct'].mean().sort_index(ascending=False)
            ref_snr = snr_groups.index.max()
            
            for snr, acc in snr_groups.items():
                rel_dist = 10 ** ((ref_snr - snr) / 20)
                f.write(f"{snr:<10.1f} | {rel_dist:<15.2f} | {acc*100:<14.2f}%\n")
            f.write("\n")

        # 3. Sample Details
        f.write("--- DETAILED SAMPLE VIEW (Top 10 Most Confident) ---\n")
        top_10 = df.sort_values(by='confidence', ascending=False).head(10)
        f.write(f"{'Filename':<30} | {'Prediction':<10} | {'Conf.':<8}\n")
        f.write("-" * 55 + "\n")
        for _, row in top_10.iterrows():
            f.write(f"{row['filename'][:28]:<30} | {row['prediction']:<10} | {row['confidence']*100:>6.1f}%\n")

    print(f"Readable report generated: {output_txt_path}")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "vit_results.csv"
    generate_readable_report(csv_file)