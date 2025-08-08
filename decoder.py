# decoder.py
from PIL import Image

# This must be identical to the delimiter in the encoder.
DELIMITER = "00000000" * 8

def decode(image_path):
    """Extracts a hidden message from an image."""
    pass