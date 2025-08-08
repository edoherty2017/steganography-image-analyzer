# encoder.py
from PIL import Image

# A delimiter of 8 null bytes to signify the end of the message.
DELIMITER = "00000000" * 8

def message_to_binary(message):
    """Converts a string message to its binary representation with a delimiter."""
    # Convert each character in the message to its 8-bit binary representation.
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    binary_message += DELIMITER
    return binary_message

def encode(image_path, message, output_path):
    """Hides a message within an image using LSB steganography."""
    pass