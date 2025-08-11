# steganalysis.py
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_comparison(original_image_path, encoded_image_path, output_path=None):
    """
    Generates a 2x2 comparison plot and four individual plots showing both
    standard histograms and difference analysis.
    """
    try:
        original_image = Image.open(original_image_path).convert("RGB")
        encoded_image = Image.open(encoded_image_path).convert("RGB")
    except FileNotFoundError:
        print("Error: One or both image files not found.")
        return

    np_orig = np.array(original_image)
    np_enc = np.array(encoded_image)

    # --- Create the main 2x2 plot ---
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

    # Plotting data on the 2x2 grid
    plot_standard_hist(axs[0, 0], np_orig[:, :, 2].flatten(), 'blue', "1. Original Image: Standard Histogram")
    plot_standard_hist(axs[0, 1], np_enc[:, :, 2].flatten(), 'blue', "2. Encoded Image: Standard Histogram")
    plot_difference_hist(axs[1, 0], np_orig[:, :, 2].flatten(), 'purple', "3. Original Image: Difference Analysis")
    plot_difference_hist(axs[1, 1], np_enc[:, :, 2].flatten(), 'purple', "4. Encoded Image: Difference Analysis")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved main comparison plot to {output_path}")

    plt.close(fig) # Close the main figure

    # --- NEW: Generate and save four individual plots ---
    if output_path:
        base_dir = os.path.dirname(output_path)
        
        # Plot 1: Original Standard Histogram
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        plot_standard_hist(ax1, np_orig[:, :, 2].flatten(), 'blue', "Original Image: Standard Histogram (Blue Channel)")
        fig1.tight_layout()
        fig1.savefig(os.path.join(base_dir, "1_original_standard_hist.png"), dpi=150)
        plt.close(fig1)

        # Plot 2: Encoded Standard Histogram
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        plot_standard_hist(ax2, np_enc[:, :, 2].flatten(), 'blue', "Encoded Image: Standard Histogram (Blue Channel)")
        fig2.tight_layout()
        fig2.savefig(os.path.join(base_dir, "2_encoded_standard_hist.png"), dpi=150)
        plt.close(fig2)

        # Plot 3: Original Difference Analysis
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        plot_difference_hist(ax3, np_orig[:, :, 2].flatten(), 'purple', "Original Image: Difference Analysis (Blue Channel)")
        fig3.tight_layout()
        fig3.savefig(os.path.join(base_dir, "3_original_difference_analysis.png"), dpi=150)
        plt.close(fig3)

        # Plot 4: Encoded Difference Analysis
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        plot_difference_hist(ax4, np_enc[:, :, 2].flatten(), 'purple', "Encoded Image: Difference Analysis (Blue Channel)")
        fig4.tight_layout()
        fig4.savefig(os.path.join(base_dir, "4_encoded_difference_analysis.png"), dpi=150)
        plt.close(fig4)

        print("Saved 4 individual analysis plots to the 'results' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive Steganalysis Tool")
    parser.add_argument("--original", required=True, help="Path to the original image.")
    parser.add_argument("--encoded", required=True, help="Path to the encoded image.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the main 2x2 analysis plot.")
    args = parser.parse_args()
    
    analyze_comparison(args.original, args.encoded, args.output)