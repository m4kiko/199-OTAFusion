import scipy.io as sio
import numpy as np
import os

def convert_mat_to_npy(mat_file_path, output_dir='Phase1_Wired_Dataset/vit_npy_data'):
    """
    Converts MATLAB dataset to individual .npy files for ViT digestion.
    Strips the 32-sample ZC preamble to extract the 1024-sample payload.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading MATLAB dataset: {mat_file_path}")
    # Load .mat file (v7.3 files require h5py, but standard MATLAB saves work with scipy)
    try:
        data = sio.loadmat(mat_file_path)
        dataset = data['dataset']
    except Exception as e:
        print(f"Error loading .mat file: {e}. If this is a large v7.3 file, use h5py.")
        return

    # Metadata for naming
    num_frames = dataset.shape[1]
    zc_len = int(data['config'][0,0]['zcLength'][0,0])
    
    print(f"Detected {num_frames} frames. Extracting payloads...")

    for i in range(num_frames):
        # Extract complex signal
        # dataset[0,i] accesses the struct element
        full_signal = dataset[0, i]['signal'].flatten()
        
        # Strip ZC Preamble (first 32 samples)
        payload = full_signal[zc_len:]
        
        # Format as (2, 1024): Row 0 = Real, Row 1 = Imag
        # ViT expects float32
        npy_signal = np.zeros((2, 1024), dtype=np.float32)
        npy_signal[0, :] = np.real(payload).astype(np.float32)
        npy_signal[1, :] = np.imag(payload).astype(np.float32)
        
        # Create descriptive filename
        mod_name = dataset[0, i]['modulation'][0]
        snr_val = dataset[0, i]['snr'][0,0]
        frame_idx = dataset[0, i]['frameIndex'][0,0]
        
        filename = f"{mod_name}_SNR{int(snr_val)}_ID{frame_idx}.npy"
        np.save(os.path.join(output_dir, filename), npy_signal)

    print(f"Conversion complete. {num_frames} files saved to '{output_dir}'.")

if __name__ == "__main__":
    # Replace with your actual path from MATLAB output
    MAT_PATH = 'Phase1_Wired_Dataset/Phase1_Complete_Dataset.mat'
    if os.path.exists(MAT_PATH):
        convert_mat_to_npy(MAT_PATH)
    else:
        print(f"File {MAT_PATH} not found. Ensure MATLAB script has run.")