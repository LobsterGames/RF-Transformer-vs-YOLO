import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


TARGET_FILE = ROOT / "data" / "sample_data" / "npy_1d" / "val" / "0_psk_sample_0008.npy"

OUT_DIR = ROOT / "images"


iq = np.load(TARGET_FILE) 

I = iq[0, :]
Q = iq[1, :]

plt.figure(figsize=(5, 5))
plt.scatter(I, Q, alpha=0.5, s=10)
plt.title(f"{TARGET_FILE.stem} Constellation Plot")
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.grid(True)
plt.axis('equal')
plt.savefig(f"{OUT_DIR}/constellation_{TARGET_FILE.stem}.png")