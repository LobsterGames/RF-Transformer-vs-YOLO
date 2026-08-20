# RF Transformer vs. YOLO: Modulation Classification

This repository explores two distinct deep learning approaches for Radio Frequency (RF) modulation classification: a custom 1D Transformer processing raw I/Q baseband data, and a YOLOv8 object detection model processing RF spectrograms. The project currently focuses on classifying Phase Shift Keying (PSK) and Quadrature Amplitude Modulation (QAM) signals.

## Repository Structure

Based on the project layout, the repository is organized as follows:

*   **`data/sample_data/`**: Contains generated training and validation datasets, including spectrogram `images/`, YOLO `labels/`, 1D baseband tensors (`npy_1d/`), and the YOLO `dataset.yaml` configuration file.
*   **`images/`**: Stores generated constellation plot visualizations.
*   **`models/`**: Stores the trained weights for both models (`transformer.pt` and `yolo.pt`).
*   **`scripts/`**: Contains all source code for data generation, model training, and evaluation.

## Included Scripts

The `scripts/` directory contains the core pipeline:

*   **`generate_data.py`**: Synthesizes clean PSK and 16-QAM signals, applies a Root-Raised Cosine filter, and adds AWGN. It automatically generates YOLO bounding box labels, spectrogram images, and 1D `.npy` baseband tensors.
*   **`train_transformer.py`**: Defines and trains the `UpgradedRFTransformer`. This custom architecture features a dynamic 4-channel feature extractor (I, Q, Amplitude, Differential Phase) and a 1D Convolutional Stem before passing data to the self-attention layers. 
*   **`evaluate_transformer.py`**: Loads the saved `transformer.pt` checkpoint and evaluates the model's loss and accuracy on the validation `.npy` dataset.
*   **`train_yolo.py`**: Trains a YOLOv8n model on the generated spectrograms. Standard spatial and color augmentations (mosaic, mixup, flips) are explicitly disabled to preserve the physical geometry of the RF signals.
*   **`evaluate_yolo.py`**: Validates the trained YOLO model against the validation image split.
*   **`constellation_plot.py`**: Reads the 1D baseband `.npy` tensors and generates I/Q constellation scatter plots.

## Getting Started

To avoid pushing massive datasets to source control, this repository only includes a tiny sample dataset. 

1.  **Generate a full dataset**: Open `scripts/generate_data.py` and change `TOTAL_SAMPLES = 10` to your desired dataset size (e.g., 2000). Run the script to populate the `data/` folder.
2.  **Train the models**: Run `train_transformer.py` and `train_yolo.py`. The scripts will automatically save the best weights to the `models/` directory.
3.  **Evaluate**: Run `evaluate_transformer.py` and `evaluate_yolo.py` to see how the models perform against each other.
4. Additionally, you can use the pre-trained model weights in `models/` (`tansformer.pt` and `yolo.pt`)