import pandas as pd
import argparse
import os
import sys

def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="View and summarize ViT classification results from CSV.")
    parser.add_argument("target_dir", type=str, help="Path to the modulation folder containing the CSV report")
    
    # IDE Support / Default behavior
    if len(sys.argv) == 1:
        print("Usage: python view_vit_results.py <path_to_modulation_folder>")
        return

    args = parser.parse_args()
    target_dir = args.target_dir
    
    # 2. Locate CSV File
    csv_path = os.path.join(target_dir, 'HOC_Classification_Results.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: Results file not found at:\n  {csv_path}")
        print("Please run bulk_classify_vit.py first to generate the report.")
        return
        
    print(f"Loading report: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    # 3. Basic Validation
    required_cols = ['SNR', 'Is_Correct', 'Predicted_Class']
    if not all(col in df.columns for col in required_cols):
        print("Error: CSV format appears invalid. Missing required columns.")
        return

    # 4. Generate Summary
    total_frames = len(df)
    total_correct = df['Is_Correct'].sum()
    overall_acc = (total_correct / total_frames) * 100
    
    print("\n" + "="*40)
    print(f"   RESULTS SUMMARY: {os.path.basename(target_dir)}")
    print("="*40)
    print(f"Total Frames Analyzed: {total_frames}")
    print(f"Overall Accuracy:      {overall_acc:.2f}%")
    
    # 5. Group by SNR
    print("\n--- Accuracy vs SNR ---")
    # Group by SNR and calculate mean of boolean 'Is_Correct'
    snr_stats = df.groupby('SNR')['Is_Correct'].mean() * 100
    
    # Sort by SNR value
    snr_stats = snr_stats.sort_index()
    
    print(f"{'SNR (dB)':<10} | {'Accuracy (%)':<15}")
    print("-" * 30)
    for snr, acc in snr_stats.items():
        print(f"{snr:<10.1f} | {acc:<15.2f}")
        
    # 6. Confusion Analysis (Top Misclassifications & Mean Probabilities)
    # Filter only incorrect predictions
    incorrect_df = df[~df['Is_Correct']]
    
    if not incorrect_df.empty:
        print("\n--- Top Misclassifications ---")
        # Count occurrences of each wrong prediction
        confusion_counts = incorrect_df['Predicted_Class'].value_counts().head(5)
        
        for pred_class, count in confusion_counts.items():
            percentage = (count / len(incorrect_df)) * 100
            print(f"Confused as {pred_class:<8}: {count} times ({percentage:.1f}% of errors)")

        print("\n--- Mean Predicted Probabilities (on Error) ---")
        # Calculate the average probability assigned to each class when the model was wrong
        prob_cols = [col for col in df.columns if col.startswith('Prob_')]
        if prob_cols:
            mean_probs = incorrect_df[prob_cols].mean()
            # Sort by probability value
            mean_probs = mean_probs.sort_values(ascending=False)
            
            for col_name, prob in mean_probs.items():
                mod_name = col_name.replace('Prob_', '')
                print(f"{mod_name:<10}: {prob:.4f}")
        else:
            print("(Probability columns not found in CSV)")

    else:
        print("\n[PERFECT] No misclassifications found!")

    print("="*40)

if __name__ == "__main__":
    main()