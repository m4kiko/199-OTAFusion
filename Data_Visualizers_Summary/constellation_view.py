import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import glob

# --- CONFIGURATION ---
# Path to your 8PSK NPY_Payload folder
INPUT_DIR = r'D:\w\Documents\199\Captured Signals\BPSK\NPY_Payload'
MOD_TYPE = '8PSK'

# ═══════════════════════════════════════════════════════════════════════════
# PREVIEW & EXPORT MODES
# ═══════════════════════════════════════════════════════════════════════════

LIVE_PREVIEW_MODE = True  # True = Fast preview on screen, False = Slower for GIF quality
SAVE_GIF = False  # Set to True to save animation as GIF

# Animation Speed (milliseconds between frames)
LIVE_FRAME_INTERVAL = 50   # Fast speed for live preview (50ms = 20fps)
GIF_FRAME_INTERVAL = 100   # Slower for smoother GIF recording

# Visualization Options
DEROTATE = True  

# GIF Export Options
GIF_OUTPUT_PATH = 'constellation_animation.gif'  # Output filename
GIF_FPS = 10  # Frames per second for GIF (lower = slower, smaller file)
GIF_DPI = 100  # Resolution (lower = smaller file, faster processing)

# --- GENERATE REFERENCE SIGNAL (Python) ---
def generate_reference_8psk(n_samples=1024):
    """Generates an ideal 8PSK signal in Python for comparison."""
    M = 8
    phase_step = 2 * np.pi / M
    # Random symbols 0 to 7
    data = np.random.randint(0, M, n_samples)
    # Modulation with 0 offset (standard)
    sig = np.exp(1j * phase_step * data)
    return sig

def update_plot(frame_idx, files, sc_captured, title_text, ax_hist, hist_im):
    """Update function for the animation."""
    if frame_idx >= len(files):
        return sc_captured,

    fpath = files[frame_idx]
    try:
        # Load signal: shape (2, 1024)
        signal = np.load(fpath)
        sig_complex = signal[0, :] + 1j * signal[1, :]
        
        # 1. Normalize Power
        pwr = np.mean(np.abs(sig_complex)**2)
        if pwr > 0:
            sig_complex = sig_complex / np.sqrt(pwr)
            
        # 2. Optional De-rotation
        if DEROTATE:
            M = 8 
            phase_est = np.angle(np.mean(sig_complex**M)) / M
            sig_complex = sig_complex * np.exp(-1j * phase_est)

        I = np.real(sig_complex)
        Q = np.imag(sig_complex)

        # Update scatter plot data
        data = np.column_stack((I, Q))
        sc_captured.set_offsets(data)
        
        # Update Histogram
        H, _, _ = np.histogram2d(I, Q, bins=100, range=[[-2, 2], [-2, 2]])
        hist_im.set_data(H.T) 
        
        # Parse SNR
        base_name = os.path.basename(fpath)
        snr_str = "Unknown"
        if "SNR_" in base_name:
            try:
                snr_str = base_name.split('SNR_')[1].split('dB')[0] + " dB"
            except:
                pass
                
        title_text.set_text(f"{MOD_TYPE} Analysis\nFile: {base_name}\nSNR: {snr_str}")
        
    except Exception as e:
        print(f"Error reading {fpath}: {e}")

    return sc_captured, title_text, hist_im

def main():
    # --- PLOT REFERENCE 8PSK (STATIC) ---
    print("Generating Python Reference 8PSK...")
    ref_sig = generate_reference_8psk()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(ref_sig), np.imag(ref_sig), s=20, c='green', marker='x', label='Python Reference')
    plt.title(f"Ideal {MOD_TYPE} (Python Generated)")
    plt.xlabel("I"); plt.ylabel("Q")
    plt.axis('equal'); plt.grid(True); plt.legend()
    plt.xlim([-1.5, 1.5]); plt.ylim([-1.5, 1.5])
    plt.show(block=False) # Show non-blocking so animation runs
    
    # --- ANIMATE CAPTURED DATA ---
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Directory '{INPUT_DIR}' not found.")
        return

    all_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.npy')))
    signal_files = [f for f in all_files if not (f.endswith('_LABEL.npy') or f.endswith('_SNR.npy'))]

    if not signal_files:
        print("No signal files found in the directory.")
        return

    print(f"Found {len(signal_files)} captured frames to animate.")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Panel 1: Scatter
    ax1.set_xlim(-2, 2); ax1.set_ylim(-2, 2)
    ax1.set_xlabel("In-Phase (I)"); ax1.set_ylabel("Quadrature (Q)")
    ax1.grid(True, alpha=0.3); ax1.set_aspect('equal')
    ax1.set_title("Captured Signal (Scatter)")
    sc_captured = ax1.scatter([], [], s=5, alpha=0.5, c='blue') 
    
    # Panel 2: Density
    ax2.set_xlim(-2, 2); ax2.set_ylim(-2, 2)
    ax2.set_xlabel("In-Phase (I)"); ax2.set_ylabel("Quadrature (Q)")
    ax2.set_aspect('equal'); ax2.set_title("Captured Signal (Density)")
    
    H = np.zeros((100, 100))
    hist_im = ax2.imshow(H.T, interpolation='nearest', origin='lower', 
                         extent=[-2, 2, -2, 2], cmap='inferno', vmin=0, vmax=50)
    
    title_text = fig.suptitle(f"{MOD_TYPE} Constellation Analysis")

    # Select frame interval based on mode
    if LIVE_PREVIEW_MODE:
        frame_interval = LIVE_FRAME_INTERVAL
        print(f"▶ Live Preview Mode: Fast playback ({LIVE_FRAME_INTERVAL}ms per frame)")
    else:
        frame_interval = GIF_FRAME_INTERVAL
        print(f"▶ Recording Mode: Slower playback for better GIF quality ({GIF_FRAME_INTERVAL}ms per frame)")

    ani = animation.FuncAnimation(
        fig, 
        update_plot, 
        frames=len(signal_files), 
        fargs=(signal_files, sc_captured, title_text, ax2, hist_im),
        interval=frame_interval, 
        blit=False, 
        repeat=True
    )

    # --- SAVE AS GIF ---
    if SAVE_GIF:
        print(f"\n{'='*60}")
        print(f"💾 SAVING ANIMATION AS GIF")
        print(f"{'='*60}")
        print(f"Output: {GIF_OUTPUT_PATH}")
        print(f"Settings: {GIF_FPS} fps, {GIF_DPI} dpi")
        print("This may take a while depending on the number of frames...")
        
        try:
            # Using Pillow writer (works without ffmpeg)
            writer = animation.PillowWriter(fps=GIF_FPS)
            ani.save(GIF_OUTPUT_PATH, writer=writer, dpi=GIF_DPI)
            print(f"\n✓ Animation saved successfully to: {GIF_OUTPUT_PATH}")
            
            # Get file size
            file_size_mb = os.path.getsize(GIF_OUTPUT_PATH) / (1024 * 1024)
            print(f"  File size: {file_size_mb:.2f} MB")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\n✗ Error saving GIF: {e}")
            print("\nTip: If you get an error, try:")
            print("  - Reducing GIF_DPI (currently {})".format(GIF_DPI))
            print("  - Reducing GIF_FPS (currently {})".format(GIF_FPS))
            print("  - Processing fewer frames")
            print(f"{'='*60}\n")
    else:
        print(f"\nℹ️  GIF saving disabled (SAVE_GIF = False)")
        print("   Set SAVE_GIF = True to export animation\n")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()