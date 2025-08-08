# decoder.py
from PIL import Image

DELIMITER = "00000000" * 8

def decode(image_path):
    """Extracts a hidden message from an image."""
    try:
        image = Image.open(image_path, 'r').convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return

    binary_data = ""
    pixels = image.load()
    width, height = image.size

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            # Extract the LSB from each color channel and append to the binary string
            binary_data += str(r & 1)
            binary_data += str(g & 1)
            binary_data += str(b & 1)
    
    # Logic to process the binary_data will be added next.