import os
import shutil

def organize_npy_files(source_dir='Phase1_Wired_Dataset/vit_npy_data'):
    """
    Scans a directory of .npy files and organizes them into subfolders
    based on the modulation name found in the filename (e.g., BPSK_SNR10_ID1.npy).
    """
    if not os.path.exists(source_dir):
        print(f"Error: Directory '{source_dir}' not found.")
        return

    # List all files in the directory
    files = [f for f in os.listdir(source_dir) if f.endswith('.npy') and os.path.isfile(os.path.join(source_dir, f))]
    
    if not files:
        print(f"No .npy files found in '{source_dir}' to organize.")
        return

    print(f"Found {len(files)} files. Organizing into subfolders...")

    count = 0
    for filename in files:
        # Extract modulation name from filename (e.g., "BPSK" from "BPSK_SNR10_ID1.npy")
        try:
            # Assumes format MOD_SNR##_ID##.npy
            mod_name = filename.split('_SNR')[0]
        except Exception:
            print(f"  Skipping {filename}: Filename format not recognized.")
            continue

        # Create target subfolder path
        target_folder = os.path.join(source_dir, mod_name)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            print(f"  Created folder: {mod_name}/")

        # Move the file
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_folder, filename)
        
        try:
            shutil.move(source_path, target_path)
            count += 1
        except Exception as e:
            print(f"  Error moving {filename}: {e}")

    print(f"\nOrganization complete. {count} files moved into subfolders within '{source_dir}'.")

if __name__ == "__main__":
    # Point this to the folder containing your flat list of .npy files
    TARGET_FOLDER = 'Phase1_Wired_Dataset/vit_npy_data'
    
    organize_npy_files(TARGET_FOLDER)