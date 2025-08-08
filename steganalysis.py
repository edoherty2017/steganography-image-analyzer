# steganalysis.py
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def analyze(image_path, output_path=None):
    """Analyzes an image and generates color histograms."""
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return

    np_image = np.array(image)
    
    r_channel = np_image[:, :, 0].flatten()
    g_channel = np_image[:, :, 1].flatten()
    b_channel = np_image[:, :, 2].flatten()
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle(f'Color Histograms for {image_path}', fontsize=16)

    ax1.hist(r_channel, bins=256, color='red', alpha=0.7)
    ax1.set_title('Red Channel')
    ax1.set_xlim([0, 255])

    ax2.hist(g_channel, bins=256, color='green', alpha=0.7)
    ax2.set_title('Green Channel')
    ax2.set_xlim([0, 255])

    ax3.hist(b_channel, bins=256, color='blue', alpha=0.7)
    ax3.set_title('Blue Channel')
    ax3.set_xlim([0, 255])

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # If an output path is provided, save the figure. Otherwise, display it.
    if output_path:
        plt.savefig(output_path)
        print(f"Histogram saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Histogram Analysis Tool")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image.")
    parser.add_argument("-o", "--output", help="Optional: Path to save the histogram plot image.")
    args = parser.parse_args()
    
    analyze(image_path=args.image, output_path=args.output)