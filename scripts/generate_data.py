import os
import cv2
import numpy as np
from scipy.signal import stft, butter, filtfilt
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

# Output directory structure
OUT_DIR = ROOT / "data" / "sample_data"

FS = 10e6           # 10 MHz
CHUNK_SIZE = 524288 # Wideband frame size
NPERSEG = 1024
TOTAL_SAMPLES = 10 # 80/20 split

os.makedirs(f"{OUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/images/val", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/val", exist_ok=True)
os.makedirs(f"{OUT_DIR}/npy_1d/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/npy_1d/val", exist_ok=True)

CLASSES = ['psk', 'qam']
CLASS_MAP = {name: i for i, name in enumerate(CLASSES)}


def apply_rrc_filter(iq_bb, sps, alpha=0.35):
    """Applies Root-Raised Cosine pulse shaping."""
    num_taps = max(31, int(sps * 6) | 1)
    t = np.arange(-num_taps // 2, num_taps // 2 + 1) / float(sps)
    h = np.zeros(len(t))
    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 - alpha + (4 * alpha / np.pi)
        elif alpha != 0 and np.isclose(abs(ti), 1.0 / (4 * alpha), atol=1e-6):
            h[i] = (alpha / np.sqrt(2)) * ((1 + 2/np.pi) * np.sin(np.pi/(4*alpha)) + (1 - 2/np.pi) * np.cos(np.pi/(4*alpha)))
        else:
            denom = np.pi * ti * (1 - (4 * alpha * ti)**2)
            num = np.sin(np.pi * ti * (1 - alpha)) + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
            h[i] = num / (denom + 1e-12)
    h /= np.sum(h) + 1e-12
    return np.convolve(iq_bb, h, mode='same')

def synthesize_clean_signal(class_name, duration_samples=16384, fs=10e6):
    """Synthesizes pure PSK or 16-QAM without fading or CFO."""
    bw = np.random.uniform(300e3, 800e3)
    sps = max(4, int(fs / bw))
    n_syms = duration_samples // sps
    
    if class_name == 'psk': # QPSK
        phases = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2], size=n_syms)
        syms = np.exp(1j * phases)
    else: # 16-QAM
        grid = np.array([-3, -1, 1, 3])
        raw_syms = np.random.choice(grid, size=n_syms) + 1j * np.random.choice(grid, size=n_syms)
        syms = raw_syms / np.sqrt(10) # Unit average power
        
    bb_upsampled = np.zeros(n_syms * sps, dtype=np.complex64)
    bb_upsampled[::sps] = syms
    iq_bb = apply_rrc_filter(bb_upsampled, sps=sps)
    
    if len(iq_bb) < duration_samples:
        iq_bb = np.pad(iq_bb, (0, duration_samples - len(iq_bb)))
    else:
        iq_bb = iq_bb[:duration_samples]
    
    fc = np.random.uniform(-3.5e6, 3.5e6)
    t = np.arange(len(iq_bb)) / fs
    rf_signal = iq_bb * np.exp(2j * np.pi * fc * t)
    
    return rf_signal, fc, bw

def extract_baseband_slice_clean(raw_slice_padded, fc, bw, fs=10e6, target_len=1024):
    """Downconverts with ZERO CFO and trims LPF edge transients."""
    N = len(raw_slice_padded)
    t = np.arange(N) / fs
    
    bb = raw_slice_padded * np.exp(-2j * np.pi * fc * t)
    cutoff = min(bw / 2.0, (fs / 2.0) - 100e3)
    b, a = butter(4, cutoff / (fs / 2.0), btype='low')
    
    real_f = filtfilt(b, a, np.real(bb))
    imag_f = filtfilt(b, a, np.imag(bb))
    filtered_bb = real_f + 1j * imag_f
    
    pad = (N - target_len) // 2
    clean_bb = filtered_bb[pad : pad + target_len] if pad > 0 else filtered_bb[:target_len]
    
    std_val = np.std(clean_bb) + 1e-8
    return (clean_bb - np.mean(clean_bb)) / std_val

print("[INIT] Generating Dataset with YOLO Labels & 1D Baseband Tensors...")

for idx in range(TOTAL_SAMPLES):
    stage = "train" if idx < (0.8 * TOTAL_SAMPLES) else "val"
    target_class = "psk" if (idx % 2 == 0) else "qam"
    class_id = CLASS_MAP[target_class]
    
    # 1. Synthesize target signal
    dur = 16384
    sig_iq, fc_val, bw_val = synthesize_clean_signal(target_class, duration_samples=dur, fs=FS)
    
    # 2. Add mild AWGN (+15 dB SNR)
    sig_pwr = np.mean(np.abs(sig_iq)**2) + 1e-12
    noise_pwr = sig_pwr / (10 ** (15.0 / 10.0))
    noise = (np.random.normal(0, np.sqrt(noise_pwr/2), dur) + 1j * np.random.normal(0, np.sqrt(noise_pwr/2), dur))
    noisy_iq = sig_iq + noise
    
    # 3. Create Wideband Frame
    frame = np.random.normal(0, np.sqrt(noise_pwr/2), CHUNK_SIZE) + 1j * np.random.normal(0, np.sqrt(noise_pwr/2), CHUNK_SIZE)
    st_sample = 100000
    frame[st_sample:st_sample+dur] += noisy_iq
    
    # 4. Save Spectrogram Image
    _, _, stft_mat = stft(frame, nperseg=NPERSEG, noverlap=NPERSEG-256, window='blackmanharris', return_onesided=False)
    log_spec = 10 * np.log10(np.abs(np.fft.fftshift(stft_mat, axes=0))**2 + 1e-12)
    norm_spec = np.clip((log_spec - np.percentile(log_spec, 10)) / 40.0 * 255.0, 0, 255).astype(np.uint8)
    
    sample_name = f"sample_{idx:04d}_{target_class}"
    cv2.imwrite(f"{OUT_DIR}/images/{stage}/{sample_name}.jpg", norm_spec)
    
    # 5. Compute Normalized YOLO Bounding Box Coordinates [0.0, 1.0]
    x_center = (st_sample + (dur / 2.0)) / CHUNK_SIZE
    width = dur / CHUNK_SIZE
    y_center = (fc_val / FS) + 0.5  # Map fc from [-FS/2, +FS/2] to [0.0, 1.0]
    height = bw_val / FS
    
    # Write YOLO annotation file: <class_id> <x_center> <y_center> <width> <height>
    with open(f"{OUT_DIR}/labels/{stage}/{sample_name}.txt", "w") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # 6. Extract Padded 1D Baseband Tensor
    pad = 64
    slice_raw = noisy_iq[2000 - pad : 2000 + 1024 + pad]
    clean_bb = extract_baseband_slice_clean(slice_raw, fc=fc_val, bw=bw_val, fs=FS, target_len=1024)
    
    tensor_2ch = np.vstack((np.real(clean_bb), np.imag(clean_bb))).astype(np.float32)
    np.save(f"{OUT_DIR}/npy_1d/{stage}/{class_id}_{target_class}_{sample_name[:-4]}.npy", tensor_2ch)

# Generate dataset.yaml for YOLO training
yaml_content = f"path: {os.path.abspath(OUT_DIR)}\ntrain: images/train\nval: images/val\n\nnames:\n  0: psk\n  1: qam\n"
with open(f"{OUT_DIR}/dataset.yaml", "w") as f:
    f.write(yaml_content)

print(f"[SUCCESS] Dataset & YOLO labels created successfully in {OUT_DIR}!")