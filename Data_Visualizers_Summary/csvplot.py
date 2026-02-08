import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION - CHANGE THESE LINES TO SWITCH BETWEEN CLASSIFIERS
# ═══════════════════════════════════════════════════════════════════════════

CLASSIFIER_NAME = 'HOC'  # Change to: 'ViT', 'Cyclo', 'HOC', 'Cyclostationary', etc.
BASE_DIR = r'D:\w\Documents\199\Captured Signals'
OUTPUT_DIR = r'D:\w\Documents\199'

# Modulation classes to analyze
MODULATION_CLASSES = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM']

# Plot styling
PLOT_STYLE = {
    'figsize': (12, 8),
    'dpi': 300,
    'grid': True,
    'marker_size': 8,
    'line_width': 2,
    'alpha': 0.7
}

# ═══════════════════════════════════════════════════════════════════════════

def extract_snr_from_filename(filename):
    """
    Extract SNR value from filename.
    Handles formats like: mod_64QAM_gain_0dB_SNR_-15.7dB_frame_07.npy
    """
    # Pattern to match SNR_<value>dB
    pattern = r'SNR_([-+]?\d+\.?\d*)dB'
    match = re.search(pattern, filename)
    
    if match:
        return float(match.group(1))
    
    return None

def load_classification_results(base_dir, modulation_class, classifier_name):
    """
    Load classification results CSV for a specific modulation class.
    Handles different CSV formats for different classifiers.
    """
    csv_filename = f'{classifier_name}_Classification_Results.csv'
    csv_path = os.path.join(base_dir, modulation_class, csv_filename)
    
    if not os.path.exists(csv_path):
        print(f"⚠ Warning: File not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        # Standardize column names based on classifier
        if classifier_name in ['Cyclo', 'HOC', 'Cyclostationary']:
            # Cyclo/HOC format: Correct, GroundTruth, Predicted, FrameIdx
            column_mapping = {
                'Correct': 'Is_Correct',
                'GroundTruth': 'Ground_Truth',
                'Predicted': 'Predicted_Class',
                'FrameIdx': 'Frame_Index'
            }
            df = df.rename(columns=column_mapping)
        # ViT format already matches: Is_Correct, Ground_Truth, Predicted_Class, Frame_Index
        
        print(f"✓ Loaded {len(df)} samples from {modulation_class}")
        return df
    except Exception as e:
        print(f"✗ Error loading {csv_path}: {e}")
        return None

def calculate_accuracy_vs_snr(df, modulation_class):
    """
    Calculate accuracy for each SNR value.
    For QPSK: Extract from filename (column is inaccurate)
    For others: Use the SNR column
    """
    if modulation_class == 'QPSK':
        # QPSK: Extract SNR from filename (column is inaccurate)
        df['SNR_Extracted'] = df['Filename'].apply(extract_snr_from_filename)
        print(f"  Note: Using SNR from filename for QPSK (column is inaccurate)")
    else:
        # Other modulations: Use SNR column
        df['SNR_Extracted'] = df['SNR']
        print(f"  Note: Using SNR from column for {modulation_class}")
    
    # Remove rows where SNR couldn't be extracted
    df_clean = df[df['SNR_Extracted'].notna()].copy()
    
    if len(df_clean) == 0:
        print(f"⚠ Warning: No valid SNR values found for {modulation_class}")
        return None
    
    # Group by SNR and calculate accuracy
    snr_groups = df_clean.groupby('SNR_Extracted')
    
    results = []
    for snr, group in snr_groups:
        total = len(group)
        correct = group['Is_Correct'].sum()
        accuracy = (correct / total) * 100
        
        results.append({
            'SNR': snr,
            'Accuracy': accuracy,
            'Total_Samples': total,
            'Correct_Samples': correct
        })
    
    results_df = pd.DataFrame(results).sort_values('SNR')
    return results_df

def plot_accuracy_vs_snr(results_dict, classifier_name, output_dir):
    """
    Create and save the accuracy vs SNR plot.
    """
    plt.figure(figsize=PLOT_STYLE['figsize'], dpi=PLOT_STYLE['dpi'])
    
    # Color map for different modulation schemes
    colors = {
        'BPSK': '#1f77b4',    # Blue
        'QPSK': '#ff7f0e',    # Orange
        '8PSK': '#2ca02c',    # Green
        '16QAM': '#d62728',   # Red
        '64QAM': '#9467bd'    # Purple
    }
    
    # Markers for variety
    markers = {
        'BPSK': 'o',
        'QPSK': 's',
        '8PSK': '^',
        '16QAM': 'D',
        '64QAM': 'v'
    }
    
    # Plot each modulation class
    for mod_class, results_df in results_dict.items():
        if results_df is not None and len(results_df) > 0:
            plt.plot(results_df['SNR'], 
                    results_df['Accuracy'],
                    marker=markers.get(mod_class, 'o'),
                    color=colors.get(mod_class, None),
                    linewidth=PLOT_STYLE['line_width'],
                    markersize=PLOT_STYLE['marker_size'],
                    alpha=PLOT_STYLE['alpha'],
                    label=mod_class)
    
    # Styling
    plt.xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title(f'{classifier_name} Classifier - Accuracy vs SNR', 
             fontsize=16, fontweight='bold', pad=20)
    
    if PLOT_STYLE['grid']:
        plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.legend(loc='lower right', fontsize=12, framealpha=0.9)
    
    # Set y-axis limits
    plt.ylim(0, 105)
    
    # Add horizontal line at 100% for reference
    plt.axhline(y=100, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    # Save the plot
    output_filename = f'{classifier_name}_Accuracy_vs_SNR.png'
    output_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_path, dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    
    # Also show the plot
    plt.show()

def print_summary_statistics(results_dict, classifier_name):
    """
    Print summary statistics for each modulation class.
    """
    print(f"\n{'='*70}")
    print(f"SUMMARY STATISTICS - {classifier_name} Classifier")
    print(f"{'='*70}\n")
    
    for mod_class, results_df in results_dict.items():
        if results_df is not None and len(results_df) > 0:
            print(f"{mod_class}:")
            print(f"  SNR Range: {results_df['SNR'].min():.1f} dB to {results_df['SNR'].max():.1f} dB")
            print(f"  Best Accuracy: {results_df['Accuracy'].max():.2f}% at {results_df.loc[results_df['Accuracy'].idxmax(), 'SNR']:.1f} dB")
            print(f"  Worst Accuracy: {results_df['Accuracy'].min():.2f}% at {results_df.loc[results_df['Accuracy'].idxmin(), 'SNR']:.1f} dB")
            print(f"  Mean Accuracy: {results_df['Accuracy'].mean():.2f}%")
            print(f"  Total Data Points: {results_df['Total_Samples'].sum()}")
            print()

def main():
    """
    Main execution function.
    """
    print(f"\n{'='*70}")
    print(f"Modulation Classification Analysis - {CLASSIFIER_NAME}")
    print(f"{'='*70}\n")
    
    # Check if directories exist
    if not os.path.exists(BASE_DIR):
        print(f"✗ Error: Base directory not found: {BASE_DIR}")
        return
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"✗ Error: Output directory not found: {OUTPUT_DIR}")
        return
    
    # Load and process data for each modulation class
    results_dict = {}
    
    for mod_class in MODULATION_CLASSES:
        print(f"\nProcessing {mod_class}...")
        df = load_classification_results(BASE_DIR, mod_class, CLASSIFIER_NAME)
        
        if df is not None:
            results_df = calculate_accuracy_vs_snr(df, mod_class)
            results_dict[mod_class] = results_df
            
            if results_df is not None:
                print(f"  Found {len(results_df)} unique SNR values")
        else:
            results_dict[mod_class] = None
    
    # Check if we have any valid results
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    
    if not valid_results:
        print("\n✗ Error: No valid data found for any modulation class!")
        return
    
    # Print summary statistics
    print_summary_statistics(results_dict, CLASSIFIER_NAME)
    
    # Create and save plot
    print(f"\nGenerating plot...")
    plot_accuracy_vs_snr(results_dict, CLASSIFIER_NAME, OUTPUT_DIR)
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()