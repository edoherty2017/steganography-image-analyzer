
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_comparison(original_image_path, encoded_image_path, output_path=None):
    """
    Generates a 2x2 comparison plot showing both standard histograms and
    difference analysis for an original and encoded image.
    """
    try:
        original_image = Image.open(original_image_path).convert("RGB")
        encoded_image = Image.open(encoded_image_path).convert("RGB")
    except FileNotFoundError:
        print("Error: One or both image files not found.")
        return

    np_orig = np.array(original_image)
    np_enc = np.array(encoded_image)

    fig, axs = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle("Comprehensive Steganalysis: Before vs. After (Blue Channel)", fontsize=24)

    def plot_standard_hist(ax, channel_data, color, title):
        ax.hist(channel_data, bins=256, color=color, alpha=0.7)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Pixel Value (0-255)", fontsize=12)
        ax.set_ylabel("Frequency (Number of Pixels)", fontsize=12)
        ax.set_xlim([0, 255])

    def plot_difference_hist(ax, channel_data, color, title):
        hist, _ = np.histogram(channel_data, bins=256, range=(0, 256))
        differences = [float(hist[i]) - float(hist[i+1]) for i in range(0, 255, 2)]
        ax.hist(differences, bins=50, color=color, alpha=0.7)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Difference between Adjacent Color Pair Counts", fontsize=12)
        ax.set_ylabel("Frequency of that Difference", fontsize=12)

    # --- Plotting Data (using Blue channel for clarity) ---
    plot_standard_hist(axs[0, 0], np_orig[:, :, 2].flatten(), 'blue', "1. Original Image: Standard Histogram")
    plot_standard_hist(axs[0, 1], np_enc[:, :, 2].flatten(), 'blue', "2. Encoded Image: Standard Histogram")
    plot_difference_hist(axs[1, 0], np_orig[:, :, 2].flatten(), 'purple', "3. Original Image: Difference Analysis")
    plot_difference_hist(axs[1, 1], np_enc[:, :, 2].flatten(), 'purple', "4. Encoded Image: Difference Analysis (The 'Smoking Gun')")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive Steganalysis Tool")
    parser.add_argument("--original", required=True, help="Path to the original image.")
    parser.add_argument("--encoded", required=True, help="Path to the encoded image.")
    parser.add_argument("-o", "--output", help="Optional: Path to save the analysis plot.")
    args = parser.parse_args()
    
    analyze_comparison(args.original, args.encoded, args.output)